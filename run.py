import openvino_genai
from openvino_genai import LLMPipeline

class ChatModel:
    """
    A simplified, non-streaming chatbot focused on getting a complete response,
    now supporting both single-turn and chat modes, and conversation refresh.
    """
    def __init__(self, model_path: str, device: str = "AUTO", is_chat_mode: bool = True):
        print(f"Loading model on device: {device}...")

        pipeline_args = {
        'models_path': model_path,
        'device': device,
        }
        # Only include the argument if the device is not CPU
        if device.upper() == "NPU":
            pipeline_args['collect_perf_metrics'] = True
            print("Performance metrics collection is enabled.")
        else:
            # If you still want a variable to track if performance metrics are collected:
            # You can set an internal flag, but the pipe object will not have it enabled.
            print("Performance metrics collection is NOT enabled for CPU / GPU (unsupported).")

        self.pipe = LLMPipeline(**pipeline_args)
        
        self.config = openvino_genai.GenerationConfig()
        self.config.max_new_tokens = 1024
        
        self.is_chat_mode = is_chat_mode
        self.history = [] # Stores conversation history for chat mode
        
        print(f"Model loaded and ready. Mode: {'Chat' if is_chat_mode else 'Single-turn'}.")
        
    def _format_performance_metrics(self, perf_metrics_obj) -> dict:
        """Helper method to format the performance metrics object."""
        if not perf_metrics_obj:
            return None
            
        return {
            'ttft (s)': round(perf_metrics_obj.get_ttft().mean / 1000, 2),
            'tpot (ms)': round(perf_metrics_obj.get_tpot().mean, 2),
            'throughput (tokens/s)': round(perf_metrics_obj.get_throughput().mean, 2),
            'new_tokens': perf_metrics_obj.get_num_generated_tokens(),
        }

    def reset_history(self):
        """
        Clears the conversation history and resets the performance metrics.
        This allows the user to start a fresh chat session.
        """
        if self.is_chat_mode:
            self.history = []
            # Resetting performance metrics in pipeline
            if self.pipe.collect_perf_metrics:
                self.pipe.reset_perf_metrics()
            print("\n[History cleared. Starting a new conversation.]")
        else:
            # Although history is not used, reset is handled gracefully
            print("\n[History is not maintained in Single-turn mode.]")

    def generate(self, prompt: str) -> dict:
        """
        Generates a full response in a single, blocking call.
        """
        if self.is_chat_mode:
            # Append only the user's new prompt if in chat mode
            self.history.append(prompt)
            input_sequence = self.history
        else:
            # In single-turn mode, we only pass the current prompt
            input_sequence = [prompt] 
        
        # We make a simple, non-streaming call to generate.
        result_obj = self.pipe.generate(input_sequence, self.config)
        
        response_text = result_obj.texts[0]
        performance_metrics = self._format_performance_metrics(result_obj.perf_metrics)
        
        # Clean up any residual 'thinking' structure from the model output
        if '</think>' in response_text:
            response_text = response_text.split('</think>')[-1].strip()
        
        if self.is_chat_mode:
            # Append the model's response to maintain conversation flow
            self.history.append(response_text)

        return {
            "text": response_text,
            "performance": performance_metrics
        }

if __name__ == "__main__":
    model_name = None
    
    #Model Selection
    while True:
        model_input=input("Please choose model to use in number \n[1] Qwen3 8b / [2] gpt-j 6b / [3] Phi-3.5 mini / [4] Deepseek-R1 available\n>>>>").lower()
        if "1" == model_input:
            model_name = "Qwen3-8B"
            break
        elif "2" == model_input:
            model_name = "gpt-j-6b"
            break
        elif "3" == model_input:
            model_name = "Phi-3.5-mini-instruct-int4"
            break
        elif "4" == model_input:
            model_name = "DeepSeek-R1-Distill-Qwen-7B"
            break
        else:
            print("Please choose among the choices")

    MODEL_PATH = f"./NPU/models/{model_name}"

    #Mode Selection
    is_chat = None
    processor = None
    while is_chat is None:
        mode_input = input("Choose mode: 'chat' (maintains history) or 'single' (no history)\n>>>>").lower()
        if mode_input in ["chat", "c"]:
            is_chat = True
        elif mode_input in ["single", "s"]:
            is_chat = False
        else:
            print("Invalid input. Please type 'chat' or 'single'.")
    
    while processor is None:
        processor_input = input("Choose mode: CPU / GPU / NPU\n>>>>").lower()
        if processor_input in ["cpu", "c"]:
            processor = "CPU"
        elif processor_input in ["gpu", "g"]:
            processor = "GPU"
        elif processor_input in ["npu", "n"]:
            processor = "NPU"
        else:
            print("Invalid input. Please type 'chat' or 'single'.")
    
    
    #Model Initialization
    model = ChatModel(model_path=MODEL_PATH, device=processor, is_chat_mode=is_chat)
    
    model_alias = model_name.split("-")[0]
    print(f"\n--- {model_alias} Chatbot Ready ({'Chat Mode' if is_chat else 'Single Mode'}) ---")
    print("Type 'exit' or 'quit' to end the session.")
    if is_chat:
        print("Type 'clear' or 'new' to start a fresh conversation and reset metrics.")
    
    #Main Chat Loop
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        
        #clear chat history
        if is_chat and user_input.lower() in ["clear", "new"]:
            model.reset_history()
            continue
        
        #skip empty input
        if not user_input.strip():
            continue
            
        response_data = model.generate(user_input)
        bot_text = response_data["text"]
        perf_metrics = response_data["performance"]

        print(f"{model_alias}: {bot_text}")
        
        if perf_metrics:
            print("\n--- Performance ---")
            key_width = max(len(key) for key in perf_metrics.keys())
            for key, value in perf_metrics.items():
                print(f"  {key.ljust(key_width)} : {value}")
            print("-------------------")

    print("\nSession ended.")