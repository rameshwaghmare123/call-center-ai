from azure.core.exceptions import (
    HttpResponseError,
    ResourceExistsError,
    ResourceNotFoundError,
    ServiceRequestError,
    ServiceResponseError,
)
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes.aio import SearchIndexClient
from azure.search.documents.indexes.models import (
    AzureOpenAIParameters,
    AzureOpenAIVectorizer,
    HnswAlgorithmConfiguration,
    LexicalAnalyzerName,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from azure.search.documents.models import (
    HybridCountAndFacetMode,
    HybridSearch,
    QueryLanguage,
    QueryType,
    ScoringStatistics,
    SearchMode,
    VectorizableTextQuery,
)
from pydantic import TypeAdapter, ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from app.helpers.config_models.ai_search import AiSearchModel
from app.helpers.http import azure_transport
from app.helpers.identity import credential
from app.helpers.logging import logger
from app.models.readiness import ReadinessEnum
from app.models.training import TrainingModel
from app.persistence.icache import ICache
from app.persistence.isearch import ISearch


class AiSearchSearch(ISearch):
    _client: SearchClient | None = None
    _config: AiSearchModel

    def __init__(self, cache: ICache, config: AiSearchModel):
        super().__init__(cache)
        logger.info("Using AI Search %s with index %s", config.endpoint, config.index)
        logger.info(
            "Note: At ~300 chars /doc, each LLM call will use approx %d tokens (without tools)",
            300 * config.top_n_documents * config.expansion_n_messages / 4,
        )
        self._config = config

    async def areadiness(self) -> ReadinessEnum:
        """
        Check the readiness of the AI Search service.
        """
        try:
            async with await self._use_client() as client:
                await client.get_document_count()
            return ReadinessEnum.OK
        except HttpResponseError:
            logger.error("Error requesting AI Search", exc_info=True)
        except ServiceRequestError:
            logger.error("Error connecting to AI Search", exc_info=True)
        except Exception:
            logger.error(
                "Unknown error while checking AI Search readiness", exc_info=True
            )
        return ReadinessEnum.FAIL

    @retry(
        reraise=True,
        retry=retry_if_exception_type(ServiceResponseError),
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=0.8, max=8),
    )
    async def training_asearch_all(
        self,
        lang: str,
        text: str,
        cache_only: bool = False,
    ) -> list[TrainingModel] | None:
        logger.debug('Searching training data for "%s"', text)
        if not text:
            return None

        # Try cache
        cache_key = f"{self.__class__.__name__}-training_asearch_all-v2-{text}"  # Cache sort method has been updated in v6, thus the v2
        cached = await self._cache.aget(cache_key)
        if cached:
            try:
                return TypeAdapter(list[TrainingModel]).validate_json(cached)
            except ValidationError as e:
                logger.debug("Parsing error: %s", e.errors())

        if cache_only:
            return None

        # Try live
        trainings: list[TrainingModel] = []
        try:
            async with await self._use_client() as client:
                results = await client.search(
                    # Full text search
                    query_language=QueryLanguage(lang.lower()),
                    query_type=QueryType.SEMANTIC,
                    search_mode=SearchMode.ANY,  # Any of the terms will match
                    search_text=text,
                    semantic_configuration_name=self._config.semantic_configuration,
                    # Vector search
                    vector_queries=[
                        VectorizableTextQuery(
                            fields="vectors",
                            text=text,
                        )
                    ],
                    # Hybrid search (full text + vector search)
                    hybrid_search=HybridSearch(
                        count_and_facet_mode=HybridCountAndFacetMode.COUNT_RETRIEVABLE_RESULTS,
                        max_text_recall_size=1000,
                    ),
                    # Relability
                    semantic_max_wait_in_milliseconds=750,  # Timeout in ms
                    # Return fields
                    include_total_count=False,  # Total count is not used
                    query_caption_highlight_enabled=False,  # Highlighting is not used
                    scoring_statistics=ScoringStatistics.GLOBAL,  # Evaluate scores in the backend for more accurate values
                    top=self._config.top_n_documents,
                )
                async for result in results:
                    try:
                        trainings.append(
                            TrainingModel.model_validate(
                                {
                                    **result,
                                    "score": (
                                        (result["@search.reranker_score"] / 4 * 5)
                                        if "@search.reranker_score" in result
                                        else (result["@search.score"] * 5)
                                    ),  # Normalize score to 0-5, failback to search score if reranker is not available
                                }
                            )
                        )
                    except ValidationError as e:
                        logger.debug("Parsing error: %s", e.errors())
        except ResourceNotFoundError:
            logger.warning('AI Search index "%s" not found', self._config.index)
        except HttpResponseError as e:
            logger.error("Error requesting AI Search: %s", e)
        except ServiceRequestError as e:
            logger.error("Error connecting to AI Search: %s", e)

        # Update cache
        if trainings:
            await self._cache.aset(
                key=cache_key,
                ttl_sec=60 * 60 * 24,  # 1 day
                value=TypeAdapter(list[TrainingModel]).dump_json(trainings),
            )

        return trainings or None

    async def _use_client(self) -> SearchClient:
        """
        Get the search client.

        If the index does not exist, it will be created.
        """
        if self._client:
            return self._client

        # Index configuration
        fields = [
            # Required field for indexing key
            SimpleField(
                name="id",
                key=True,
                type=SearchFieldDataType.String,
            ),
            # Custom fields
            SearchableField(
                analyzer_name=LexicalAnalyzerName.STANDARD_LUCENE,
                name="content",
                type=SearchFieldDataType.String,
            ),
            SearchableField(
                analyzer_name=LexicalAnalyzerName.STANDARD_LUCENE,
                name="title",
                type=SearchFieldDataType.String,
            ),
            SearchField(
                name="vectors",
                searchable=True,
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                vector_search_dimensions=self._config.embedding_dimensions,
                vector_search_profile_name="profile-default",
            ),
        ]
        vector_search = VectorSearch(
            profiles=[
                VectorSearchProfile(
                    algorithm_configuration_name="algorithm-default",
                    name="profile-default",
                    vectorizer="vectorizer-default",
                ),
            ],
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="algorithm-default",
                ),
            ],
            vectorizers=[
                AzureOpenAIVectorizer(
                    name="vectorizer-default",
                    # Without credentials specified, the database will use its system managed identity
                    azure_open_ai_parameters=AzureOpenAIParameters(
                        deployment_id=self._config.embedding_deployment,
                        model_name=self._config.embedding_model,
                        resource_uri=self._config.embedding_endpoint,
                    ),
                )
            ],
        )
        semantic_search = SemanticSearch(
            default_configuration_name=self._config.semantic_configuration,
            configurations=[
                SemanticConfiguration(
                    name=self._config.semantic_configuration,
                    prioritized_fields=SemanticPrioritizedFields(
                        title_field=SemanticField(
                            field_name="title",
                        ),
                        content_fields=[
                            SemanticField(
                                field_name="content",
                            ),
                        ],
                    ),
                ),
            ],
        )

        # Create index if it does not exist
        async with SearchIndexClient(
            # Deployment
            endpoint=self._config.endpoint,
            index_name=self._config.index,
            # Index configuration
            fields=fields,
            semantic_search=semantic_search,
            vector_search=vector_search,
            # Performance
            transport=await azure_transport(),
            # Authentication
            credential=await credential(),
        ) as client:
            try:
                await client.create_index(
                    SearchIndex(
                        fields=fields,
                        name=self._config.index,
                        vector_search=vector_search,
                    )
                )
                logger.info('Created Search "%s"', self._config.index)
            except ResourceExistsError:
                pass
            except HttpResponseError as e:
                if not e.error or not e.error.code == "ResourceNameAlreadyInUse":
                    raise e

        # Return client
        self._client = SearchClient(
            # Deployment
            endpoint=self._config.endpoint,
            index_name=self._config.index,
            # Performance
            transport=await azure_transport(),
            # Authentication
            credential=await credential(),
        )
        return self._client
