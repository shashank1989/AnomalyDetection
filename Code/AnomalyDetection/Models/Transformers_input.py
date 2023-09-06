import torch
def get_input_data_and_attention_mask(x_train_test_type,tokenizer,device):

    # Set batch size and max sequence length
    batch_size = 16
    max_seq_length = 512

    # Create empty lists for input data
    input_ids_list = []
    attention_mask_list = []

    # Process sequences in batches
    for i in range(0, len(x_train_test_type), batch_size):
        batch_sequences = x_train_test_type[i:i+batch_size]
        batch_texts = [' '.join(str(num) for num in sequence) for sequence in batch_sequences]

        # Tokenize batch_texts all at once
        inputs = tokenizer(
            batch_texts,
            padding='max_length',
            truncation=True,
            max_length=max_seq_length,
            return_tensors='pt',
            return_attention_mask=True
        ).to(device)

        input_ids_list.append(inputs['input_ids'])
        attention_mask_list.append(inputs['attention_mask'])

        # Clear GPU memory to manage memory consumption
        if i % 1000 == 0:
            torch.cuda.empty_cache()

    return input_ids_list,attention_mask_list




