"""
Utility functions for SFT data generation
"""

from openai import AsyncOpenAI


async def async_chat(base_url, api_key, **openai_args):
    """
    Async chat completion using OpenAI-compatible API

    Args:
        base_url: API base URL
        api_key: API key
        **openai_args: Additional arguments for chat completion (model, messages, max_tokens, temperature, etc.)

    Returns:
        Response text if successful, None otherwise
    """
    async_client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=300,
        max_retries=0,
    )
    try:
        completion = await async_client.chat.completions.create(**openai_args)
        res = completion.choices[0].message.content
        if res is not None:
            return res
        return None
    except Exception as e:
        return None
