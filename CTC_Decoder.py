import numpy as np

def ctc_decoder(predictions, chars, length):
    # Initialize variables
    input_seq_len = len(predictions[0])  # Length of the predicted sequence
    blank_label = len(chars)  # Index of the blank label

    # Initialize a list to store the decoded texts
    decoded_texts = []

    for pred_seq in predictions:
        # Initialize a list to store the decoded text for the current sequence
        decoded_text = []

        # Initialize the previous label to be the blank label
        prev_label = blank_label

        # Iterate through the predicted labels for the sequence
        for label_list in pred_seq:
            # Ensure label_list is a list of integers
            label_list = [int(label) for label in label_list]

            # Skip consecutive duplicates and the blank label
            for label in label_list:
                if label != prev_label and label != blank_label:
                    decoded_text.append(chars[label])

                # Update the previous label
                prev_label = label

        # Combine consecutive duplicates and create a final decoded text
        decoded_texts.append(''.join(decoded_text))
    decoded_texts = decoded_texts[0][:length]
    return decoded_texts

# Example usage:
predicted_labels = [[[2.3, 3, 4]]]
character_set = ['a','b','c','d','e']

decoded_labels = ctc_decoder(predicted_labels, character_set,1)
print(decoded_labels)
decoded_labels = ctc_decoder(predicted_labels, character_set,2)
print(decoded_labels)
decoded_labels = ctc_decoder(predicted_labels, character_set,3)
print(decoded_labels)
# print(char_list)
