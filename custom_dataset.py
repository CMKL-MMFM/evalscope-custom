import os
import pandas as pd
import numpy as np
from vlmeval.dataset.image_base import ImageBaseDataset
from vlmeval.dataset.image_vqa import CustomVQADataset
from vlmeval.smp import load, dump, d2df


class CustomDataset:

    def load_data(self, dataset):
        data_path = os.path.join(
            os.path.expanduser("~/LMUData"),
            f"{dataset}.tsv"
        )

        df = pd.read_csv(data_path, sep="\t")

        def clean_base64(x):
            if isinstance(x, str) and x.startswith("data:image"):
                return x.split(",", 1)[1]
            return x

        df["image"] = df["image"].apply(clean_base64)
        df["index"] = df["index"].astype(str)

        return df

    def build_prompt(self, line):
        # Strip prefix from line before parent processes it
        # because dump_image inside parent reads line['image'] directly
        if isinstance(line.get('image'), str) and line['image'].startswith("data:image"):
            line = line.copy()
            line['image'] = line['image'].split(",", 1)[1]

        msgs = ImageBaseDataset.build_prompt(self, line)

        # Only append choices when at least one option has text (e.g. TGAT has all-empty choices)
        choices = [str(line.get(k, '') or '') for k in ['A', 'B', 'C', 'D']]
        if any(c.strip() for c in choices):
            choice_text = f"\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}"
            msgs[-1]['value'] += "\n" + choice_text

        msgs[-1]['value'] += "\nAnswer with A, B, C, or D only."

        return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        data = load(eval_file)
        assert 'answer' in data and 'prediction' in data

        print(data[['answer', 'prediction']])

        data['prediction'] = [str(x) for x in data['prediction']]
        data['answer'] = [str(x) for x in data['answer']]

        result = np.mean(data['answer'] == data['prediction'])
        ret = {'Overall': result}
        ret = d2df(ret).round(2)

        suffix = eval_file.split('.')[-1]
        result_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(ret, result_file)

        return ret


CustomVQADataset.load_data = CustomDataset.load_data
CustomVQADataset.build_prompt = CustomDataset.build_prompt
CustomVQADataset.evaluate = CustomDataset.evaluate