import torch
import gc
from tqdm import tqdm
from datasets import load_dataset

# ------------------------------
# ActivationBuffer using real model
# ------------------------------
class ActivationBuffer:
    """
    A buffer that stores activations extracted from a model.
    The activations are stored as a tensor of shape:
      (n_ctxs, ctx_len, d_submodule)
    and __next__() yields a batch of contexts (activations) with that shape.
    
    When the number of unread contexts falls below half the buffer,
    the refresh() method is called. Here, refresh() uses the real model
    to compute activations (via run_with_cache) from a batch of text data.
    """
    def __init__(
        self, 
        data,  # generator yielding text data
        model,  # the HookedSAETransformer (gemma-2-2b)
        submodule,  # not used here (but kept for API compatibility)
        d_submodule,  # activation dimension (e.g. 2304)
        io='out',  # either 'in' or 'out' (unused in our demo)
        n_ctxs=100,  # total contexts stored in the buffer
        ctx_len=256,  # sequence length per context
        refresh_batch_size=16,  # number of texts processed per refresh step
        out_batch_size=8,  # number of contexts yielded per __next__ call
        device='cpu', 
        remove_bos: bool = False,
    ):
        if io not in ['in', 'out']:
            raise ValueError("io must be either 'in' or 'out'")
        
        # Get the model's data type from one of its parameters.
        self.model_dtype = next(model.parameters()).dtype
        
        self.d_submodule = d_submodule
        self.n_ctxs = n_ctxs
        self.ctx_len = ctx_len
        # Buffer of activations: (n_ctxs, ctx_len, d_submodule)
        self.activations = torch.empty(n_ctxs, ctx_len, d_submodule, device=device, dtype=self.model_dtype)
        # One flag per context (True if already yielded)
        self.read = torch.zeros(n_ctxs, dtype=torch.bool, device=device)
        self.data = data
        self.model = model
        self.submodule = submodule  # not used here
        self.io = io
        self.refresh_batch_size = refresh_batch_size
        self.out_batch_size = out_batch_size
        self.device = device
        self.remove_bos = remove_bos
    
    def __iter__(self):
        return self

    def __next__(self):
        # If fewer than half of the contexts remain unread, refresh the buffer.
        if (~self.read).sum() < self.n_ctxs // 2:
            self.refresh()

        # Get indices of unread contexts
        unread_idxs = (~self.read).nonzero(as_tuple=False).squeeze()
        if unread_idxs.dim() == 0:
            unread_idxs = unread_idxs.unsqueeze(0)
        # Randomly select a batch of unread contexts
        perm = torch.randperm(len(unread_idxs), device=unread_idxs.device)
        batch_size = min(self.out_batch_size, len(unread_idxs))
        idxs = unread_idxs[perm[:batch_size]]
        self.read[idxs] = True
        return self.activations[idxs]  # shape: (batch_size, ctx_len, d_submodule)
    
    def text_batch(self, batch_size=None):
        """
        Return a list of texts from the data generator.
        """
        if batch_size is None:
            batch_size = self.refresh_batch_size
        texts = []
        try:
            for _ in range(batch_size):
                texts.append(next(self.data))
        except StopIteration:
            if texts:
                return texts
            else:
                raise StopIteration("End of data stream reached")
        return texts
    
    def refresh(self):
        """
        Refresh the buffer by retaining unread activations and
        filling the remainder by running the model on new text.
        """
        gc.collect()
        torch.cuda.empty_cache()
        # Retain unread contexts.
        unread_acts = self.activations[~self.read]
        current_count = unread_acts.shape[0]
        new_activations = torch.empty(self.n_ctxs, self.ctx_len, self.d_submodule, 
                                        device=self.device, dtype=self.model_dtype)
        if current_count > 0:
            new_activations[:current_count] = unread_acts
        current_idx = current_count
        
        # Fill the rest of the buffer using refresh_batch_size texts per iteration.
        while current_idx < self.n_ctxs:
            texts = self.text_batch(self.refresh_batch_size)
            # Tokenize the texts using the model's built-in tokenizer.
            tokens = self.model.to_tokens(texts)
            # Truncate to ctx_len; if needed, pad with zeros.
            if tokens.shape[1] < self.ctx_len:
                padding = torch.zeros((tokens.shape[0], self.ctx_len - tokens.shape[1]),
                                      dtype=tokens.dtype, device=tokens.device)
                tokens = torch.cat([tokens, padding], dim=1)
            else:
                tokens = tokens[:, :self.ctx_len]
            tokens = tokens.to(self.device)
            # Run the model and capture activations from the desired layer.
            with torch.no_grad():
                output, cache = self.model.run_with_cache(
                    tokens,
                    names_filter=lambda name: name == "blocks.20.hook_resid_post"
                )
            acts = cache["blocks.20.hook_resid_post"]  # shape: (batch, ctx_len, d_submodule)
            batch_count = acts.size(0)
            remaining = self.n_ctxs - current_idx
            if batch_count > remaining:
                acts = acts[:remaining]
                batch_count = remaining
            new_activations[current_idx: current_idx+batch_count] = acts
            current_idx += batch_count
        
        self.activations = new_activations
        self.read = torch.zeros(self.n_ctxs, dtype=torch.bool, device=self.device)
    
    @property
    def config(self):
        return {
            'd_submodule': self.d_submodule,
            'io': self.io,
            'n_ctxs': self.n_ctxs,
            'ctx_len': self.ctx_len,
            'refresh_batch_size': self.refresh_batch_size,
            'out_batch_size': self.out_batch_size,
            'device': self.device
        }

# ------------------------------
# Helper: Create a generator from a HF dataset
# ------------------------------
def hf_dataset_to_generator(dataset_name, split="train", streaming=True):
    """
    Creates a generator from a Hugging Face dataset. Each iteration yields the 'text' field.
    """
    dataset = load_dataset(dataset_name, split=split, streaming=streaming)
    def gen():
        for x in dataset:
            yield x["text"]
    return gen()

# ------------------------------
# Main: Load the real gemma-2-2b model and test the ActivationBuffer
# ------------------------------
def main():
    # Use CUDA if available.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load a streaming dataset (using the medical-qa combined dataset).
    data_generator = hf_dataset_to_generator("charlieoneill/medical-qa-combined", split="train", streaming=True)
    
    # Import and load the gemma-2-2b model (HookedSAETransformer).
    from sae_lens import HookedSAETransformer
    print("Loading gemma-2-2b model...")
    model_hooked = HookedSAETransformer.from_pretrained("gemma-2-2b", device=device)
    torch.cuda.empty_cache()
    
    # Set parameters: sequence length and activation dimension.
    ctx_len = 256        # maximum sequence length
    activation_dim = 2304  # gemma-2-2b's model dimension
    
    # Create the ActivationBuffer.
    # Here n_ctxs determines how many contexts are stored overall,
    # refresh_batch_size is how many texts are processed at a time,
    # and out_batch_size is the batch size yielded when iterating.
    buffer = ActivationBuffer(
        data=data_generator,
        model=model_hooked,
        submodule=None,  # not used here
        d_submodule=activation_dim,
        n_ctxs=100,       # for demonstration; adjust as needed
        ctx_len=ctx_len,
        refresh_batch_size=16,
        out_batch_size=8,
        device=device,
        remove_bos=False
    )
    
    # Initial fill of the buffer.
    print("Refreshing activation buffer...")
    buffer.refresh()
    
    # Test: retrieve a few batches from the buffer and print their shapes.
    print("Testing ActivationBuffer with gemma-2-2b activations...")
    for i in range(3):
        batch = next(buffer)
        # Each batch should have shape (out_batch_size, ctx_len, activation_dim)
        print(f"Batch {i+1} shape: {batch.shape}")
    print("ActivationBuffer test completed.")

if __name__ == "__main__":
    main()