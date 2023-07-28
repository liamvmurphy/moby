from typing import Any, Dict, List, Optional

from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from llm.models.FunctionCall import FunctionCall
from llm.models.OpenAiAnswer import OpenAiAnswer
from logger import get_logger
from models.chat import ChatHistory
from repository.chat.get_chat_history import get_chat_history
from repository.chat.update_chat_history import update_chat_history
from supabase.client import Client, create_client
from vectorstore.supabase import CustomSupabaseVectorStore

from .base import BaseBrainPicking

logger = get_logger(__name__)


def format_answer(model_response: Dict[str, Any]) -> OpenAiAnswer:
    answer = model_response["choices"][0]["message"]
    content = answer["content"]
    function_call = None

    if answer.get("function_call", None) is not None:
        function_call = FunctionCall(
            answer["function_call"]["name"],
            answer["function_call"]["arguments"],
        )

    return OpenAiAnswer(
        content=content,
        function_call=function_call,  # pyright: ignore reportPrivateUsage=none
    )


class OpenAIFunctionsBrainPicking(BaseBrainPicking):
    """
    Class for the OpenAI Brain Picking functionality using OpenAI Functions.
    It allows to initialize a Chat model, generate questions and retrieve answers using ConversationalRetrievalChain.
    """

    # Default class attributes
    model: str = "gpt-3.5-turbo-0613"

    def __init__(
        self,
        model: str,
        chat_id: str,
        temperature: float,
        max_tokens: int,
        brain_id: str,
        user_openai_api_key: str,
        # TODO: add streaming
    ) -> "OpenAIFunctionsBrainPicking":  # pyright: ignore reportPrivateUsage=none
        super().__init__(
            model=model,
            chat_id=chat_id,
            max_tokens=max_tokens,
            user_openai_api_key=user_openai_api_key,
            temperature=temperature,
            brain_id=str(brain_id),
            streaming=False,
        )

    @property
    def openai_client(self) -> ChatOpenAI:
        return ChatOpenAI(
            openai_api_key=self.openai_api_key
        )  # pyright: ignore reportPrivateUsage=none

    @property
    def embeddings(self) -> OpenAIEmbeddings:
        return OpenAIEmbeddings(
            openai_api_key=self.openai_api_key
        )  # pyright: ignore reportPrivateUsage=none

    @property
    def supabase_client(self) -> Client:
        return create_client(
            self.brain_settings.supabase_url, self.brain_settings.supabase_service_key
        )

    @property
    def vector_store(self) -> CustomSupabaseVectorStore:
        return CustomSupabaseVectorStore(
            self.supabase_client,
            self.embeddings,
            table_name="vectors",
            brain_id=self.brain_id,
        )

    def _get_model_response(
        self,
        messages: List[Dict[str, str]],
        functions: Optional[List[Dict[str, Any]]] = None,
    ) -> Any:
        """
        Retrieve a model response given messages and functions
        """
        logger.info("Getting model response")
        kwargs = {
            "messages": messages,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if functions:
            logger.info("Adding functions to model response")
            kwargs["functions"] = functions

        return self.openai_client.completion_with_retry(**kwargs)

    def _get_chat_history(self) -> List[Dict[str, str]]:
        """
        Retrieves the chat history in a formatted list
        """
        history = get_chat_history(self.chat_id)

        # Convert the chat history into the appropriate list
        formatted_history = [
            item
            for chat in history
            for item in [
                {"role": "user", "content": chat.user_message},
                {"role": "assistant", "content": chat.assistant},
            ]
        ]

        # Filter for pairs where the user message starts with "This is a reminder"
        reminder_pairs = [
            pair
            for pair in zip(formatted_history[::2], formatted_history[1::2])
            if pair[0]['content'].startswith("This is a reminder")
        ]

        # Extract the last reminder pair, if any
        extracted_message = reminder_pairs[-1] if reminder_pairs else None
        logger.info("3 Getting chat history")
        # Get only the last 6 items (last 3 pairs of user and assistant messages)
        last_three_pairs = formatted_history[-6:]

        # If there was an extracted message, add it to the start of the last_three_pairs list
        if extracted_message is not None:
            final_list = list(extracted_message) + last_three_pairs
        else:
            final_list = last_three_pairs



        for i, item in enumerate(final_list):
            logger.info(f"Message {i + 1}: {item}")

        return final_list

    def _get_context(self, question: str) -> str:
        """
        Retrieve documents related to the question
        """
        logger.info("Getting context")
        logger.info(f"Question: {question}")
        return self.vector_store.similarity_search(
            query=question
        )  # pyright: ignore reportPrivateUsage=none

    def _construct_prompt(
        self, question: str, useContext: bool = False, useHistory: bool = False
    ) -> List[Dict[str, str]]:
        """
        Constructs a prompt given a question, and optionally include context and history
        """
        logger.info("Constructing prompt")
        system_messages = [
            {
                "role": "system",
                "content": """Your name is Moby. You are an helpful Chemistry, Mathematics, Biology and English Teacher for grades 1-12 that has access to a person's documents and that can answer questions about them.
                Answer all questions like you are talking to a 10 year old.
                You will not answer questions that do not relate to Chemistry, Mathematics, Biology or English and you will not answer questions that are additionally not related to grades 1-12.
                Write the answer in the same language as the question.
                You have access to functions to help you answer the question.
                If you don't know the answer, just say that you don't know but be helpful and explain why you can't answer""",
            }
        ]

        if useHistory:
            logger.info("Adding chat history to prompt")
            history = self._get_chat_history()
            system_messages.append(
                {"role": "system", "content": "Previous messages are already in chat."}
            )
            system_messages.extend(history)

        if useContext:
            logger.info("Adding chat context to prompt")
            chat_context = self._get_context(question)
            context_message = f"Here are the documents you have access to: {chat_context if chat_context else 'No document found'}"
            system_messages.append({"role": "user", "content": context_message})

        system_messages.append({"role": "user", "content": question})

        return system_messages

    def generate_answer(self, question: str) -> ChatHistory:
        """
        Main function to get an answer for the given question
        """
        logger.info("Getting answer")
        functions = [
            {
                "name": "get_history_and_context",
                "description": "Get the chat history between you and the user and also get the relevant documents to answer the question. Always use that unless a very simple question is asked that a 5 years old could answer or if the user says continue or something like that.",
                "parameters": {"type": "object", "properties": {}},
            },
        ]

        # First, try to get an answer using just the question
        # response = self._get_model_response(
        #     messages=self._construct_prompt(question), functions=functions
        # )
        # formatted_response = format_answer(response)

        # If the model calls for history, try again with history included
        if (
            True
            # formatted_response.function_call
            # and formatted_response.function_call.name == "get_history"
        ):
            logger.info("Model called for history")
            response = self._get_model_response(
                messages=self._construct_prompt(question, useHistory=True),
                functions=[],
            )

            formatted_response = format_answer(response)

        if (
            formatted_response.function_call
            and formatted_response.function_call.name == "get_history_and_context"
        ):
            logger.info("Model called for history and context")
            response = self._get_model_response(
                messages=self._construct_prompt(
                    question, useContext=True, useHistory=True
                ),
                functions=[],
            )
            formatted_response = format_answer(response)

        # Update chat history
        chat_history = update_chat_history(
            chat_id=self.chat_id,
            user_message=question,
            assistant=formatted_response.content or "",
        )

        return chat_history
