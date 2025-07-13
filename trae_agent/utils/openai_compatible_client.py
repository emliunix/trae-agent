# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""OpenAI API client wrapper with tool integration."""

import json
from typing import override

import openai
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
    ChatCompletion,
    ChatCompletionMessage,
)
from openai.types.shared_params import FunctionDefinition
from ..tools.base import Tool, ToolCall, ToolResult
from ..utils.config import ModelParameters
from .base_client import BaseLLMClient
from .llm_basics import LLMMessage, LLMResponse, LLMUsage
from .retry_utils import retry_with


class OpenAIClient(BaseLLMClient):
    """OpenAI client wrapper with tool schema generation."""

    def __init__(self, model_parameters: ModelParameters):
        super().__init__(model_parameters)

        self.client: openai.OpenAI = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.message_history: list[ChatCompletionMessageParam] = []

    @override
    def set_chat_history(self, messages: list[LLMMessage]) -> None:
        """Set the chat history."""
        self.message_history = self.parse_messages(messages)

    def _create_openai_response(
        self,
        api_call_input: list[ChatCompletionMessageParam],
        model_parameters: ModelParameters,
        tool_schemas: list[ChatCompletionToolParam] | None,
    ) -> ChatCompletion:
        """Create a response using OpenAI API. This method will be decorated with retry logic."""
        return self.client.chat.completions.create(
            messages=api_call_input,
            tools=tool_schemas if tool_schemas else openai.NOT_GIVEN,
            model=model_parameters.model,
            temperature=model_parameters.temperature,
            top_p=model_parameters.top_p,
            max_tokens=model_parameters.max_tokens,
        )

    @override
    def chat(
        self,
        messages: list[LLMMessage],
        model_parameters: ModelParameters,
        tools: list[Tool] | None = None,
        reuse_history: bool = True,
    ) -> LLMResponse:
        """Send chat messages to OpenAI with optional tool support."""
        openai_messages: list[ChatCompletionMessageParam] = self.parse_messages(messages)

        tool_schemas = None
        if tools:
            tool_schemas = [
                ChatCompletionToolParam(
                    function=FunctionDefinition(
                        name=tool.name,
                        description=tool.description,
                        parameters=tool.get_input_schema(),
                        strict=True,
                    ),
                    type="function",
                )
                for tool in tools
            ]

        api_call_input: list[ChatCompletionMessageParam] = []
        if reuse_history:
            api_call_input.extend(self.message_history)
        api_call_input.extend(openai_messages)

        # Apply retry decorator to the API call
        retry_decorator = retry_with(
            func=self._create_openai_response,
            max_retries=model_parameters.max_retries,
        )
        response = retry_decorator(api_call_input, model_parameters, tool_schemas)
        output = response.choices[0].message

        parsed_output = self.parse_response(output)
        if parsed_output:
            self.message_history = api_call_input + [parsed_output]

        content = ""
        tool_calls: list[ToolCall] = []
        for tool_call in output.tool_calls or []:
            tool_calls.append(
                ToolCall(
                    call_id=tool_call.id,
                    name=tool_call.function.name,
                    arguments=json.loads(tool_call.function.arguments)
                    if tool_call.function.arguments
                    else {},
                    id=tool_call.id,
                )
            )
        if output.content:
            content = output.content

        usage = None
        if response.usage:
            usage = LLMUsage(
                input_tokens=response.usage.prompt_tokens or 0,
                output_tokens=response.usage.completion_tokens or 0,
                cache_read_input_tokens=response.usage.prompt_tokens_details.cached_tokens or 0
                if response.usage.prompt_tokens_details else 0,
                reasoning_tokens=response.usage.completion_tokens_details.reasoning_tokens or 0
                if response.usage.completion_tokens_details else 0,
            )

        llm_response = LLMResponse(
            content=content,
            usage=usage,
            model=response.model,
            finish_reason=response.choices[0].finish_reason,
            tool_calls=tool_calls if len(tool_calls) > 0 else None,
        )

        # Record trajectory if recorder is available
        if self.trajectory_recorder:
            self.trajectory_recorder.record_llm_interaction(
                messages=messages,
                response=llm_response,
                provider="openai_compatible",
                model=model_parameters.model,
                tools=tools,
            )

        return llm_response

    @override
    def supports_tool_calling(self, model_parameters: ModelParameters) -> bool:
        """Check if the current model supports tool calling."""
        return True

    def parse_messages(self, messages: list[LLMMessage]) -> list[ChatCompletionMessageParam]:
        """Parse the messages to OpenAI format."""
        openai_messages: list[ChatCompletionMessageParam] = []
        for msg in messages:
            if msg.tool_result:
                openai_messages.append(self.parse_tool_call_result(msg.tool_result))
            else:
                if not msg.content:
                    raise ValueError("Message content is required")
                if msg.role == "system":
                    openai_messages.append({"role": "system", "content": msg.content})
                elif msg.role == "user":
                    openai_messages.append({"role": "user", "content": msg.content})
                elif msg.role == "assistant":
                    openai_messages.append({"role": "assistant", "content": msg.content})
                else:
                    raise ValueError(f"Invalid message role: {msg.role}")
        return openai_messages

    def parse_tool_call_result(self, tool_call_result: ToolResult) -> ChatCompletionToolMessageParam:
        """Parse the tool call result from the LLM response to FunctionCallOutput format."""
        result_content: str = ""
        if tool_call_result.result is not None:
            result_content += str(tool_call_result.result)
        if tool_call_result.error:
            result_content += f"\nError: {tool_call_result.error}"
        result_content = result_content.strip()

        return ChatCompletionToolMessageParam(
            role="tool",  # Explicitly set the type field
            tool_call_id=tool_call_result.call_id,
            content=result_content,
        )
    
    def parse_response(self, message: ChatCompletionMessage) -> ChatCompletionMessageParam | None:
        """Parse the OpenAI response to a ChatCompletionMessageParam."""
        if message.content:
            return {
                "role": message.role,
                "content": message.content,
            }
        else:
            return None
