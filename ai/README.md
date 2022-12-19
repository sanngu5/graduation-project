## AI 실행환경
1. conda 환경 설정
```
conda env create -f environment.yml 
conda activate cpp
```

2. ffmpeg 설치
* ffmpeg (video to png)


## IQA(cal3.py) 실행환경
```
conda create -n iqa python=3.8 -y
conda activate iqa

conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch

pip install pyiqa
pip install pillow
pip install tqdm
pip install numpy
```

## data_crawler.ipynb 실행환경
moviepy의 caption.py의 xml_caption_to_srt함수를 다음과 같이 수정 후 사용

```
def xml_caption_to_srt(self, xml_captions: str) -> str:
    """Convert xml caption tracks to "SubRip Subtitle (srt)".

    :param str xml_captions:
        XML formatted caption tracks.
    """
    segments = []
    root = ElementTree.fromstring(xml_captions)
    for i, child in enumerate(list(root.findall('body/p'))):
        text = "".join(child.itertext()).strip()
        if not text:
            continue
        caption = unescape(text.replace("\n", " ").replace("  ", " "),)
        try:
            duration = float(child.attrib["d"])
        except KeyError:
            duration = 0.0
        start = float(child.attrib["t"])
        end = start + duration
        start, end = start/1000, end/1000
        sequence_number = i + 1  # convert from 0-indexed to 1.
        line = "{seq}\n{start} --> {end}\n{text}\n".format(
            seq=sequence_number,
            start=self.float_to_srt_time_format(start),
            end=self.float_to_srt_time_format(end),
            text=caption,
        )
        segments.append(line)
    return "\n".join(segments).strip()
```

## Citation
If any part of our paper and repository is helpful to your work, please generously cite with:
```
@inproceedings{chu2021deep,
  title={Deep Video Decaptioning},
  author={Chu Pengpeng, Quan Weize, Wang Tong, Wang Pan, Ren Peiran and Yan Dong-Ming},
  booktitle = {The Proceedings of the British Machine Vision Conference (BMVC)},
  year={2021}
}
```