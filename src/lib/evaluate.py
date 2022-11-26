from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import os
import warnings
from nltk.translate.bleu_score import sentence_bleu
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics import SacreBLEUScore



def generate(prompt, model_obj, device, model_save_name, gen_max_len):
    """
    Feel free to modify this function as per your requirements.
    """

    if model_obj.base_model == 'GPT2':
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2' + model_obj.model_size)
    else:
        raise Exception("Error: this library supports only GPT2 as the base model for now.")

    if not os.path.exists(model_save_name):
        warnings.warn("User provided model doesn't exist. Using default model instead.")
    else:
        state_dict = torch.load(model_save_name)
        model.load_state_dict(state_dict)

    model = model_obj.model
    model = model.to(device)
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    outputs_ = model.generate(
        input_ids,
        max_length=gen_max_len,
        do_sample=True,
        top_k=20,
        no_repeat_ngram_size=2,
        temperature=0.7
        )

    output = tokenizer.decode(outputs_[0], skip_special_tokens=True)

    return output



def evaluate(test_loader, model_obj, test_split_token, test_end_token, model_save_name, gen_max_len):
    """
    Calculate the BLEU, ROUGE, and SacreBLEU scores for the model.
    Returns a dictionary of metric scores.
    """
    print("EVALUATING" + "." * 20)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    BLEUscores = []
    ROUGEscores = []
    SacreBLEUscores = []

    rouge = ROUGEScore()
    sacre_bleu = SacreBLEUScore()

    if model_obj.base_model == 'GPT2':
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2' + model_obj.model_size)
    else:
        raise Exception("Error: this library supports only GPT2 as the base model for now.")
        
    for text in test_loader:

        text = tokenizer.decode(text[0], skip_special_tokens=True)
        if test_split_token is None:
            inp_num_words = 10
            text = text.split(" ")
            inp = " ".join(text[:inp_num_words])
            ref = text[inp_num_words:]
        else:
            text = text.split(test_split_token)
            inp = text[0]
            ref = inp + text[1].replace(test_end_token, '')

        gen = generate(inp, model_obj, device, model_save_name, gen_max_len)

        if test_end_token is None:
            ref = inp + " " + " ".join(ref[:gen_max_len])
        else:
            gen = gen.split(test_end_token)[0]
            gen = gen.replace(test_split_token, '')

        ROUGEscores.append(rouge(gen, ref)['rouge1_recall'].item())
        SacreBLEUscores.append(sacre_bleu([gen], [[ref]]).item())
        ref = [ref.split(" ")]
        gen = gen.split(" ")
        BLEUscores.append(sentence_bleu(ref, gen))

    BLEUscore = sum(BLEUscores) / len(BLEUscores)
    ROUGEscore = sum(ROUGEscores) / len(ROUGEscores)
    SacreBLEUscore = sum(SacreBLEUscores) / len(SacreBLEUscores)

    metrics = {
      'BLEU': BLEUscore,
      'ROUGE (recall)': ROUGEscore,
      'SacreBLEU': SacreBLEUscore
    }

    return metrics








