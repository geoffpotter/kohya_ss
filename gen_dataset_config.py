# This script will create the caption text files in the specified folder using the specified file pattern and caption text.
#
# eg: python caption.py D:\some\folder\location "*.png, *.jpg, *.webp" "some caption text"

import argparse
import toml
#import glob
import os
from pathlib import Path

class data_config_general:
    def __init__(self, dict=None) -> None:
        self.shuffle_caption = True
        self.caption_extension = '.captions'
        self.tag_extension = '.tags'
        self.keep_tokens = 0
        self.caption_tag_dropout_rate = 0.99
        self.class_tokens = None
        self.template_file_name = None
        if not dict is None:
            self.shuffle_caption = dict.get('shuffle_caption', self.shuffle_caption)
            self.caption_extension = dict.get('caption_extension', self.caption_extension)
            self.tag_extension = dict.get('tag_extension', self.tag_extension)
            self.keep_tokens = dict.get('keep_tokens', self.keep_tokens)
            self.caption_tag_dropout_rate = dict.get('caption_tag_dropout_rate', self.caption_tag_dropout_rate)
            self.class_tokens = dict.get('class_tokens', self.class_tokens)
            self.template_file_name = dict.get('template_file_name', self.template_file_name)
    def getDict(self):
        return {
            'shuffle_caption': self.shuffle_caption,
            'caption_extension': self.caption_extension,
            'tag_extension': self.tag_extension,
            'keep_tokens': self.keep_tokens,
            'caption_tag_dropout_rate': self.caption_tag_dropout_rate,
            'class_tokens': self.class_tokens,
            'template_file_name': self.template_file_name,
        }
    
class data_config_dataset:
    def __init__(self, dict:dict=None) -> None:
        self.shuffle_caption = None
        self.is_reg = None
        self.num_repeats = None
        self.keep_tokens = None
        self.caption_extension = None
        self.tag_extension = None
        self.caption_tag_dropout_rate = None
        self.class_tokens = None
        self.template_file_name = None
        self.x_resolution = None
        self.y_resolution = None
        self.subsets:list[subset] = []
        if not dict is None:
            self.shuffle_caption = dict.get('shuffle_caption', self.shuffle_caption)
            self.is_reg = dict.get('is_reg', self.is_reg)
            self.num_repeats = dict.get('num_repeats', self.num_repeats)
            self.keep_tokens = dict.get('keep_tokens', self.keep_tokens)
            self.caption_extension = dict.get('caption_extension', self.caption_extension)
            self.tag_extension = dict.get('tag_extension', self.tag_extension)
            self.caption_tag_dropout_rate = dict.get('caption_tag_dropout_rate', self.caption_tag_dropout_rate)
            self.class_tokens = dict.get('class_tokens', self.class_tokens)
            self.template_file_name = dict.get('template_file_name', self.template_file_name)
            self.x_resolution = dict.get('x_resolution', self.x_resolution)
            self.y_resolution = dict.get('y_resolution', self.y_resolution)
            for subset in dict.get('subsets', self.subsets):
                print("subset?", subset)
                self.subsets.append(data_config_subset(subset))
    def getDict(self):
        return {
            'shuffle_caption': self.shuffle_caption,
            'is_reg': self.is_reg,
            'num_repeats': self.num_repeats,
            'keep_tokens': self.keep_tokens,
            'caption_extension': self.caption_extension,
            'tag_extension': self.tag_extension,
            'caption_tag_dropout_rate': self.caption_tag_dropout_rate,
            'class_tokens': self.class_tokens,
            'template_file_name': self.template_file_name,
            'resolution': [self.x_resolution, self.y_resolution],
            'subsets': [subset.getDict() for subset in self.subsets]
        }

class data_config_subset:
    def __init__(self, dict:dict=None) -> None:
        self.image_dir:Path = None #'path/to/imgs'
        self.shuffle_caption = None
        self.is_reg = None
        self.num_repeats = None
        self.keep_tokens = None
        self.caption_extension = None
        self.tag_extension = None
        self.caption_tag_dropout_rate = None
        self.class_tokens = None
        self.template_file_name = None
        if not dict is None:
            self.image_dir = dict.get('image_dir', self.image_dir)
            self.shuffle_caption = dict.get('shuffle_caption', self.shuffle_caption)
            self.is_reg = dict.get('is_reg', self.is_reg)
            self.num_repeats = dict.get('num_repeats', self.num_repeats)
            self.keep_tokens = dict.get('keep_tokens', self.keep_tokens)
            self.caption_extension = dict.get('caption_extension', self.caption_extension)
            self.tag_extension = dict.get('tag_extension', self.tag_extension)
            self.caption_tag_dropout_rate = dict.get('caption_tag_dropout_rate', self.caption_tag_dropout_rate)
            self.class_tokens = dict.get('class_tokens', self.class_tokens)
            self.template_file_name = dict.get('template_file_name', self.template_file_name)
    def getDict(self):
        return {
            'image_dir': self.image_dir.absolute().as_posix(),
            'shuffle_caption': self.shuffle_caption,
            'is_reg': self.is_reg,
            'num_repeats': self.num_repeats,
            'keep_tokens': self.keep_tokens,
            'caption_extension': self.caption_extension,
            'tag_extension': self.tag_extension,
            'caption_tag_dropout_rate': self.caption_tag_dropout_rate,
            'class_tokens': self.class_tokens,
            'template_file_name': self.template_file_name
        }


class data_config:
    def __init__(self, path:Path, dict=None) -> None:
        self.general = data_config_general()
        self.datasets:list[data_config_dataset] = []
        self.path = path
        #print('loading', dict)
        if not dict is None:
            self.general = data_config_general(dict.get('general', self.general))
            for dataset in dict.get('datasets', self.datasets):
                print("dataset?", dataset)
                self.datasets.append(data_config_dataset(dataset))
    
    def getDict(self):
        return {
            'general': self.general.getDict(),
            'datasets': [ds.getDict() for ds in self.datasets]
        }

    def save(self):
        config =  self.getDict()
        # print(self.datasets)
        # print("---", self.path, toml.dumps(config), "--", self.path)
        with open(self.path, "w") as f:
            toml.dump(config, f)
        
def get_dataset_config(configs:dict[str, data_config], datasets:dict[str, data_config_dataset], concept:Path, base_folder:str, this_folder:Path):
    if not concept in configs:
        print("starting concept config:", concept)
        template_file_name = Path(os.path.join(base_folder, concept + "-512.toml"))
        # print(template_file_name)
        # template_files = template_file_name.glob("*.template")
        # templates = []
        # for idx, template in enumerate(template_files):
        #     templates.append(template)
        # print(templates)
        # for tmpl in templates:
        #     print(tmpl)
        # if not os.path.isfile(template_file_name):
        #     template_file_name = None
        if False: #template_file_name.exists():
            existing_file = toml.load(template_file_name)
            dc = data_config(template_file_name, existing_file)
        else:
            dc = data_config(template_file_name)
        dc.general.class_tokens = concept
        #dc.general.template_file_name = template_file_name
        configs[concept] = dc
    else:
        dc = configs[concept]
    
    if not concept in datasets:
        print("starting dataset config:", concept)
        # template_file_name = Path(os.path.join(base_folder, concept_class, concept))
        # templates = Path.glob(template_file_name, "*.template")
        # for idx, template in enumerate(templates):
        #     print(template)
        # for tmpl in templates:
        #     print(tmpl)
        # if not os.path.isfile(template_file_name):
        #     template_file_name = None
        ds = data_config_dataset()
        ds.class_tokens = concept
        ds.x_resolution = 512
        ds.y_resolution = 512
        ds.num_repeats = 5
        #dc.general.template_file_name = template_file_name
        datasets[concept] = ds
        dc.datasets.append(ds)
    else:
        ds = datasets[concept]
    
    
    return dc, ds
            


def create_config_files(base_folder: str, file_pattern: str, caption_file_ext: str, overwrite: bool):
    # Split the file patterns string and strip whitespace from each pattern
    patterns = [pattern.strip() for pattern in file_pattern.split(",")]

    # Create a Path object for the image folder
    folder = Path(base_folder)
    print("making config for:", folder)
    # Iterate over the file patterns
    concept_configs:dict[str, data_config] = {}
    concept_datasets:dict[str, data_config_dataset] = {}


    class_configs:dict[str, data_config] = {}
    class_datasets:dict[str, data_config_dataset] = {}

    all_config:data_config = data_config(Path(os.path.join(base_folder, "all-512.toml")))
    all_dataSet = data_config_dataset()
    all_dataSet.x_resolution = all_dataSet.y_resolution = 1024
    all_dataSet.num_repeats = 5
    all_config.datasets.append(all_dataSet)
    all_dataSet.class_tokens = "person"

    for pattern in patterns:
        #print("looking for", pattern, "in", folder)
        # Use the glob method to match the file patterns
        files = folder.glob(pattern)
        folders = list(set([file.parent for file in files]))
        #print(folders)
        for idx, this_folder in enumerate(folders):
            orig_this_folder = this_folder
            this_folder = this_folder.relative_to(base_folder)
            #concept_class = this_folder.parts[0]
            concept_class = "woman"
            concept = this_folder.parts[0]
            is_reg = this_folder.name.endswith("_reg")

            concept_config, concept_dataset = get_dataset_config(concept_configs, concept_datasets, concept, base_folder, folder)
            class_config, class_dataset = get_dataset_config(class_configs, class_datasets, concept_class, base_folder, folder)

            subset = data_config_subset()
            subset.image_dir = orig_this_folder
            if is_reg:
                subset.is_reg = is_reg
                subset.num_repeats = 1
            # else:
            #     subset.num_repeats = 5

            concept_dataset.subsets.append(subset)
            class_dataset.subsets.append(subset)
            if subset.is_reg:
                subset.class_tokens = concept_class
            else:
                subset.class_tokens = concept
            all_dataSet.subsets.append(subset)

    for name in concept_configs:
        config = concept_configs[name]
        config.save()
        #no reg
        config.path = Path(os.path.join(base_folder, config.path.stem + "-noreg.toml"))
        non_regs = []
        for subset in config.datasets[0].subsets:
            if not subset.is_reg:
                non_regs.append(subset)
        config.datasets[0].subsets = non_regs
        config.save()

    all_config.save()
    #print(all_dataSet.subsets)
        # Iterate over the matched files
        # for idx, file in enumerate(files):
        #     file_folder = file.parent

        #     # Check if a text file with the same name as the current file exists in the folder
        #     txt_file = file.with_suffix(caption_file_ext)
        #     if not txt_file.exists() or overwrite:
        #         # Create a text file with the caption text in the folder, if it does not already exist
        #         # or if the overwrite argument is True
        #         with open(txt_file, "w") as f:
        #             f.write(caption_text)



    # ds1 = data_config_dataset()
    # ds1.class_tokens = 'test'
    # dss1 = data_config_subset()
    # ds1.subsets.append(dss1)
    # out_file = os.path.join(base_folder, "test.toml")
    # existing = toml.load(out_file)

    # config = data_config(existing)
    # config.datasets.append(ds1)
    

    # config_dict =  {
    #         'general': config.general.getDict(),
    #         'datasets': [
    #             {"t":1}
    #         ]
    #     }

    # config.save(out_file)

def main():
    # Define command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_folder", type=str, help="the folder where the image files are located")
    parser.add_argument("--file_pattern", type=str, default="**/*.png, **/*.jpg, **/*.jpeg, **/*.webp", help="the pattern to match the image file names")
    parser.add_argument("--caption_file_ext", type=str, default=".caption", help="the caption file extension.")
    parser.add_argument("--overwrite", action="store_true", default=True, help="whether to overwrite existing caption files")

    # Create a mutually exclusive group for the caption_text and caption_file arguments
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--caption_text", type=str, help="the text to include in the caption files")
    group.add_argument("--caption_file", type=argparse.FileType("r"), help="the file containing the text to include in the caption files")

    # Parse the command-line arguments
    args = parser.parse_args()
    base_folder = args.base_folder
    file_pattern = args.file_pattern
    caption_file_ext = args.caption_file_ext
    overwrite = args.overwrite

    # Create a Path object for the image folder
    folder = Path(base_folder)

    # Check if the image folder exists and is a directory
    if not folder.is_dir():
        raise ValueError(f"{base_folder} is not a valid directory.")
        
    # Create the caption files
    create_config_files(base_folder, file_pattern, caption_file_ext, overwrite)

if __name__ == "__main__":
    main()