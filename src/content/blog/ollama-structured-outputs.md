---
author: Dan Clayton
pubDatetime: 2024-12-09T17:15:00
title: How Does Ollama's Structured Outputs Work?
postSlug: ollama-structured-outputs
featured: false
draft: false
description: Digging into what Ollama is doing under the hood to produce structured outputs
---

A few days ago, Ollama [released 'structured outputs' with v0.5](https://ollama.com/blog/structured-outputs). At the minute, it is limited to producing JSON formats. I've wondered for a while how structured outputs work (OpenAI and others have supported it for some time), and I thought I would find out what it is doing. It's pretty clear that being able to produce good JSON with an LLM would be helpful, and is a key step in turning them into something more useful than a chatbot.

My initial thinking was as a flaky predictor of next tokens, producing JSON would be unreliable, but something more interesting is going on. I'm fairly new to Ollama so my first port of call was the Git diff for v0.5 - it's easy to see what they've added.

### How does Ollama actually interact with models?

The first thing to understand is that Ollama uses [llama.cpp](https://github.com/ggerganov/llama.cpp) to run the inference of models. Ollama is a Golang web server that handles the loading and management of different models, and provides OpenAI schema compatible REST endpoints for interacting with models.

When a chat completion, embedding or tokenisation is received, Ollama handles that via calls to a llama.cpp server that it has spawned (listening on a random port).

> The llama.cpp server is mostly managed [through this code](https://github.com/ollama/ollama/blob/main/llm/server.go). The most useful code for working out what Ollama is doing is the [`Completion` function](https://github.com/ollama/ollama/blob/da09488fbfc437c55a94bc5374b0850d935ea09f/llm/server.go#L697).

## Grammars

Previously with ollama, you can pass a `format` parameter into the API endpoint as `json` (just the string literal).

If you set format to `json`, how does Ollama 'make' the model respond with JSON?

You can always ask the model to produce JSON in your prompt, which does work to a degree, but it can break down and be unreliable.

Instead, [llama.cpp supports the use of 'GBNF grammars'](https://github.com/ggerganov/llama.cpp/blob/master/grammars/README.md) to constrain a model output, so you can force the model to only produce valid JSON, or speak only in emoji etc. Llama cpp has actually supported grammars for a relatively long time.

BNF grammars are used to describe the syntax of programming languages - so they're ideal for describing the desired output of an LLM.

The language model produces tokens with an associated probability (logit) - llama.cpp uses the grammar to work out which tokens are valid according to the current state. Any tokens that are not valid as the next token according to the grammar are masked (forbidden) during the sampling stage of the LLM output.
In the llama.cpp [MR](https://github.com/ggerganov/llama.cpp/pull/1773/) that introduced this feature there was\* a nice block that made this functionality apparent:

<small>\* these lines have since been changed but I'm treating this as pseudocode.</small>

```cpp
// loop over tokens, if not valid then make it impossible to sample
...
    if (!valid) {
        candidates->data[i].logit = -INFINITY;
    }
```

The model still produces tokens with logits in the same way as it would normally, but when we are sampling its output we only take the tokens that would actually be valid output.

So, if the model has so far generated a `{`, it might assign a probability to `'`, `"`, `hello`, `}` as next possible tokens, but after filtering that down to valid JSON, our sampler instead picks from `"` and `}`.

When you set the `format` parameter to `json`, ollama helpfully passes an already written grammar for valid JSON. In case you're interested, [here it is hardcoded](https://github.com/ollama/ollama/blob/da09488fbfc437c55a94bc5374b0850d935ea09f/llm/server.go#L634-L654) into ollama's source. It's not trivial to understand.

### From JSON Schema to Grammar

Since ollama v0.5, instead of using generic JSON grammar built-in to ollama, you can supply an actual JSON schema, and [Ollama will generate the grammar](https://github.com/ollama/ollama/blob/da09488fbfc437c55a94bc5374b0850d935ea09f/llama/sampling_ext.cpp#L62) specifically for that JSON schema. By sending this to llama.cpp, the generated output has to conform to your JSON schema.

Ollama will generate that schema for you. You can generate the GBNF yourself with the [Python example that llama.cpp provides](https://github.com/ggerganov/llama.cpp/blob/3d98b4cb226c3140bd1ae6c65ed126b7d90332fa/examples/json_schema_to_grammar.py).

## How does this affect the model?

The model doesn't _'know'_ that there is masking of its output tokens going on using a grammar. However, once the logit manipulation has caused it to generate a few tokens of JSON, the model then has that context to help it produce the rest of its output. I think this is why it works so well. The Ollama maintainers do recommend that you include instructions in the prompt for the model to output JSON. If you don't - the token sampling might be happening a long way from the top of the distribution and you get rubbish responses, or lots of whitespace.

I [did some testing](https://gist.github.com/danclaytondev/51c6d1add250c092777c7e4a4b773341) by running Ollama in debug mode and found that the model does not see the format you supply as additional context. That is unlike [tool calling](https://ollama.com/blog/tool-support) where you supply a JSON spec for the model to call and the JSON format for that tool is injected into its system prompt by Ollama.

I think it makes sense to include the JSON format you want both into the prompt and as a schema for use in the grammar; the model can try to produce the right output and we just help it to be valid.

Ollama does not validate the full response from the model against the schema, so if the model stops producing tokens mid-JSON without closing braces etc, it won't be valid JSON despite the grammar restrictions.

It's worth mentioning that using grammar sampling can degrade the performance of inference - the generation of tokens through the LLM is optimised and runs on the GPU as far as I understand at the minute the masking of tokens is not parallelised. Depending on the grammar it could be quite slow to mask the tokens. [There is an open issue on llama.cpp for performance enhancements](https://github.com/ggerganov/llama.cpp/issues/4218).
