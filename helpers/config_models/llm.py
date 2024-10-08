from azure.core.credentials import AzureKeyCredential
from openai import AsyncAzureOpenAI
from pydantic import BaseModel
from rtclient import (
    RTClient,
)

from helpers.identity import token


class AbstractPlatformModel(BaseModel):
    """
    Shared properties for all LLM platform models.
    """

    context: int
    deployment: str
    endpoint: str
    model: str
    seed: int = 42  # Reproducible results
    streaming: bool
    temperature: float = 0.0  # Most focused and deterministic


class RealtimePlatformModel(AbstractPlatformModel):
    """
    Properties for the realtime LLM models, like `gpt-4o-realtime`.
    """

    api_key: str

    async def instance(self) -> tuple[RTClient, "RealtimePlatformModel"]:
        client = RTClient(
            # Deployment
            azure_deployment=self.deployment,
            model=self.model,
            # Authentication
            key_credential=AzureKeyCredential(self.api_key),
        )
        return client, self


class SequentialPlatformModel(AbstractPlatformModel):
    """
    Properties for the sequential LLM models, like `gpt-4o-mini`.
    """

    _client: AsyncAzureOpenAI | None = None
    api_version: str = "2024-06-01"
    deployment: str
    endpoint: str
    model: str

    async def instance(self) -> tuple[AsyncAzureOpenAI, "SequentialPlatformModel"]:
        if not self._client:
            self._client = AsyncAzureOpenAI(
                # Reliability
                max_retries=0,  # Retries are managed manually
                timeout=60,
                # Deployment
                api_version=self.api_version,
                azure_deployment=self.deployment,
                azure_endpoint=self.endpoint,
                # Authentication
                azure_ad_token_provider=await token(
                    "https://cognitiveservices.azure.com/.default"
                ),
            )
        return self._client, self


class LlmModel(BaseModel):
    """
    Properties for the LLM configuration.
    """

    realtime: RealtimePlatformModel
    sequential: SequentialPlatformModel
