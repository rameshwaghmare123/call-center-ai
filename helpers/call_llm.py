import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable

# from helpers.call_utils import
import numpy as np
from azure.communication.callautomation.aio import CallAutomationClient
from rtclient import (
    InputAudioTranscription,
    RTClient,
    RTInputItem,
    RTOutputItem,
    RTResponse,
    ServerVAD,
)
from scipy.signal import resample

from helpers.config import CONFIG
from helpers.features import phone_silence_timeout_sec
from helpers.llm_tools import LlmPlugins
from helpers.logging import logger
from helpers.monitoring import tracer
from models.call import CallStateModel

_cache = CONFIG.cache.instance()
_db = CONFIG.database.instance()


@tracer.start_as_current_span("call_load_llm_chat")
async def load_llm_chat(
    audio_length: int,
    audio_sample_rate: int,
    audio_stream: AsyncGenerator[bytes, None],
    automation_client: CallAutomationClient,
    call: CallStateModel,
    post_callback: Callable[[CallStateModel], Awaitable[None]],
    training_callback: Callable[[CallStateModel], Awaitable[None]],
) -> CallStateModel:
    # Create client
    rtc_client, llm_model = await CONFIG.llm.realtime.instance()

    # Build plugins
    plugins = LlmPlugins(
        call=call,
        client=automation_client,
        post_callback=post_callback,
    )
    tools = await plugins.to_openai(call)

    # Configure LLM
    await rtc_client.configure(
        input_audio_format="pcm16",
        input_audio_transcription=InputAudioTranscription(model="whisper-1"),
        modalities={"text"},
        temperature=0,
        tool_choice="auto",
        tools=tools,
        turn_detection=ServerVAD(silence_duration_ms=phone_silence_timeout_sec),
    )

    # Run in/out tasks
    await asyncio.gather(
        _in_audio(
            client=rtc_client,
            duration_ms=audio_length,
            sample_rate=audio_sample_rate,
            stream=audio_stream,
        ),
        _out_messages(
            automation_client=automation_client,
            call=call,
            client=rtc_client,
            post_callback=post_callback,
            training_callback=training_callback,
        ),
    )

    # Return updated call
    return call


async def _out_messages(
    automation_client: CallAutomationClient,
    call: CallStateModel,
    client: RTClient,
    post_callback: Callable[[CallStateModel], Awaitable[None]],
    training_callback: Callable[[CallStateModel], Awaitable[None]],
) -> None:
    await asyncio.gather(
        _out_messages_items(client=client),
        _out_messages_control(client=client),
    )


async def _out_messages_control(client: RTClient) -> None:
    async for control in client.control_messages():
        if control is not None:
            print(f"Received a control message: {control.type}")
        else:
            break


async def _out_item(item: RTOutputItem):
    arguments = None
    audio_transcript = None
    text_data = None

    async for chunk in item:
        if chunk.type == "audio_transcript":
            audio_transcript = (audio_transcript or "") + chunk.data
        elif chunk.type == "tool_call_arguments":
            arguments = (arguments or "") + chunk.data
        elif chunk.type == "text":
            text_data = (text_data or "") + chunk.data

    if text_data is not None:
        logger.info(f"Text: {text_data}")
    if audio_transcript is not None:
        logger.info(f"Audio Transcript: {audio_transcript}")
    if arguments is not None:
        logger.info(f"Tool Call Arguments: {arguments}")


async def _out_response(client: RTClient, response: RTResponse):
    prefix = f"[response={response.id}]"
    async for item in response:
        print(prefix, f"Received item {item.id}")
        asyncio.create_task(_out_item(item=item))
    print(prefix, "Response completed")
    await client.close()


async def _out_input_item(item: RTInputItem):
    prefix = f"[input_item={item.id}]"
    await item
    print(prefix, f"Previous Id: {item.previous_id}")
    print(prefix, f"Transcript: {item.transcript}")
    print(prefix, f"Audio Start [ms]: {item.audio_start_ms}")
    print(prefix, f"Audio End [ms]: {item.audio_end_ms}")


async def _out_messages_items(client: RTClient) -> None:
    async for item in client.items():
        if isinstance(item, RTResponse):
            asyncio.create_task(
                _out_response(
                    response=item,
                    client=client,
                )
            )
        else:
            asyncio.create_task(_out_input_item(item))


async def _in_audio(
    client: RTClient,
    duration_ms: int,
    sample_rate: int,
    stream: AsyncGenerator[bytes, None],
) -> None:
    # Resampling parameters
    target_sample_rate = 24000
    target_samples_per_chunk = sample_rate * (duration_ms / 1000)
    target_bytes_per_sample = 2
    target_bytes_per_chunk = int(target_samples_per_chunk * target_bytes_per_sample)

    # Consumes audio stream
    async for audio_bytes in stream:
        # Resample audio if necessary
        if sample_rate != target_sample_rate:
            audio_chunk = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_chunk = _resample_audio(
                data=audio_chunk,
                original_rate=sample_rate,
                target_rate=target_sample_rate,
            )
            audio_bytes = audio_chunk.tobytes()

        # Send audio in chunks
        for i in range(0, len(audio_bytes), target_bytes_per_chunk):
            chunk = audio_bytes[i : i + target_bytes_per_chunk]
            await client.send_audio(chunk)


def _resample_audio(
    data: np.ndarray,
    original_rate: int,
    target_rate: int,
) -> np.ndarray:
    samples = round(len(data) * float(target_rate) / original_rate)
    resampled = resample(x=data, num=samples)
    return resampled.astype(np.int16)
