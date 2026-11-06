// Chat conversion functions for server (Responses API, Anthropic API, OAI streaming diffs)

#pragma once

#include "chat.h"
#include "server-common.h"
#include "server-http.h"

#include "json.h"

// Convert OpenAI Responses API format to OpenAI Chat Completions API format
json server_chat_convert_responses_to_chatcmpl(const json & body);

// Convert OpenAI Responses API function and namespace tools to OpenAI Chat Completions API tools
json server_chat_convert_responses_tools(const json & tools);

std::string server_chat_encode_responses_tool_name(const std::string & tool_namespace, const std::string & name);
json server_chat_format_responses_tool_call(const common_chat_tool_call & tool_call, const std::string & status);

// Convert Anthropic Messages API format to OpenAI Chat Completions API format
json server_chat_convert_anthropic_to_oai(const json & body);

// convert OpenAI transcriptions API format to OpenAI Chat Completions API format
json convert_transcriptions_to_chatcmpl(
    const json & body,
    const common_chat_templates * tmpls,
    const std::map<std::string, uploaded_file> & in_files,
    std::vector<raw_buffer> & out_files);

// prompt of the "asr" chat template, live ASR models (e.g. LFM3-Audio live ASR) transcribe with it
// empty if the model has no such template, throws on a template error
llama_tokens server_chat_get_asr_prompt(const llama_model * model);

json server_chat_msg_diff_to_json_oaicompat(const common_chat_msg_diff & diff);
