import re


def replace_ligatures(sentence):
    replace_dic = {"ﬂ": "fl", "ﬁ": "fi", "ﬀ": "ff", "ﬅ": "ft"}
    for k, v in replace_dic.items():
        sentence = sentence.replace(k, v)
    return sentence


def clean_text(text):
    sentence = re.sub(r"\b-[^\S\r\n]*\n", "", text)  # Glue Words together
    sentence = re.sub(r"\n", " ", sentence)  # Delete line breaks
    sentence = replace_ligatures(sentence)  # Get rid of ligatures
    sentence = re.sub("[^a-zA-Z.,]+", " ", sentence)  # Only letters!
    sentence = re.sub("http", " ", sentence)
    sentence = re.sub("w{3}.[a-zA-z]+.[a-zA-z]+", " ", sentence)

    return sentence


def set_aliases(pdf_explorer_inst, pdf_files_lst):
    return {
        idx: article
        for idx, article in list(
            zip(list(pdf_explorer_inst.corpus.keys()), pdf_files_lst)
        )
    }


def get_article_name(explorer_inst, idx):
    return explorer_inst.aliases[idx]

def get_article_id(explorer_inst, article_name):
    swapped_dic = dict((v,k) for k,v in explorer_inst.aliases.items())
    return swapped_dic[article_name]


