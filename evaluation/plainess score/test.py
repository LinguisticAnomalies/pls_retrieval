from fairseq.models.roberta import RobertaModel
import torch
import argparse
from tqdm import tqdm
import pickle
from fairseq.data.data_utils import collate_tokens


@torch.no_grad()
def generate(roberta, infile, outfile="roberta_hypo.txt", bsz=32, n_obs=None, classification_head_name=None):
    count = 1
    label_fn = lambda label: roberta.task.label_dictionary.string(
    [label + roberta.task.label_dictionary.nspecial]
    )
   
    output = []
    with open(infile) as source_file:
        source = [s.strip() for s in source_file.readlines()]
        slines = []
        for i in tqdm(range(len(source))):
            # if count % 10000 == 0:
            #     break
            slines.append(source[i])   
            if (len(slines) == bsz) or (i == len(source)-1):
                # tokens = roberta.encode(slines)
                tokens = collate_tokens([roberta.encode(sent) for sent in slines], pad_idx=1)
                # pred = label_fn(roberta.predict(classification_head_name, tokens).argmax().item())
                pred = list(torch.exp(roberta.predict(classification_head_name, tokens)).cpu().detach().numpy()[:,0])
                # pred = roberta.predict(classification_head_name, tokens, return_logits=True)
                # if pred == 1:
                #     print('hahahahahahahhaha')
                # print(pred)
                # output.append(int(float(pred)))
                output += pred
                slines = [] 
    pickle.dump(output, open(outfile, "wb"))



def main():
    """
    Usage::

         python examples/bart/summarize.py \
            --model-dir $HOME/bart.large.cnn \
            --model-file model.pt \
            --src $HOME/data-bin/cnn_dm/test.source
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        required=True,
        type=str,
        default="roberta.large/",
        help="path containing model file and src_dict.txt",
    )
    parser.add_argument(
        "--data-path",
        required=True,
        type=str,
        default=None,
        help="path containing data-bin",
    )
    parser.add_argument(
        "--model-file",
        default="checkpoint_best.pt",
        help="where in model_dir are weights saved",
    )
    parser.add_argument(
        "--src", default="test.source", help="text to summarize", type=str
    )
    parser.add_argument(
        "--out", default="test.hypo", help="where to save summaries", type=str
    )
    parser.add_argument("--bsz", default=32, help="where to save summaries", type=int)
    parser.add_argument("--classification-head-name", default='sentence_classification_head', help="classification head name", type=str)
    parser.add_argument(
        "--n", default=None, help="how many examples to summarize", type=int
    )
    args = parser.parse_args()
    roberta = RobertaModel.from_pretrained(
        args.model_dir,
        checkpoint_file=args.model_file,
        data_name_or_path=args.data_path,
    )
    roberta = roberta.eval()
    if torch.cuda.is_available():
        roberta = roberta.cuda().half()
    generate(
        roberta, args.src, bsz=args.bsz, n_obs=args.n, outfile=args.out, classification_head_name=args.classification_head_name
    )


if __name__ == "__main__":
    main()
