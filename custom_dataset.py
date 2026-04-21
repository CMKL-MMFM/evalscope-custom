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

        if "image" in df.columns:
            df["image"] = df["image"].fillna("").apply(clean_base64)
            # Drop the image column when all values are empty so ImageBaseDataset
            # skips image_map processing (empty string triggers a broken index lookup)
            if df["image"].str.strip().eq("").all():
                df = df.drop(columns=["image"])
        df["index"] = df["index"].astype(str)

        return df

    def build_prompt(self, line):
        image = line.get('image', '')

        # Strip data-URI prefix before parent processes it
        if isinstance(image, str) and image.startswith("data:image"):
            line = line.copy()
            line['image'] = image = image.split(",", 1)[1]

        all_keys = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
        choices = [(k, str(line.get(k, '') or '')) for k in all_keys]
        choices = [(k, v) for k, v in choices if v.strip()]
        choice_text = ""
        if choices:
            choice_text = "\n" + "\n".join(f"{k}. {v}" for k, v in choices)
            last_key = choices[-1][0]
            choice_keys = ", ".join(k for k, _ in choices)

        # Text-only dataset — skip dump_image entirely
        if not image:
            question = str(line.get('question', ''))
            suffix = f"\nAnswer with the correct option letter ({choice_keys})." if choices else ""
            return [{'type': 'text', 'value': question + choice_text + suffix}]

        msgs = ImageBaseDataset.build_prompt(self, line)
        suffix = f"\nAnswer with the correct option letter ({choice_keys})." if choices else ""
        msgs[-1]['value'] += choice_text + suffix
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