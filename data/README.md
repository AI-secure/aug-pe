
## Dataset Preparation
Download the Yelp `train.csv` (1.21G)  and PubMed `train.csv` (117MB)  from [this link](https://drive.google.com/drive/folders/1oSICwgCAqdxEz4mF5ZK863RoN5sxMB_0?usp=sharing) or execute:
```bash 
cd aug-pe
bash scripts/download_data.sh # download yelp train.csv and pubmed train.csv
```


## Dataset Description: 
- Yelp: Processed Yelp dataset from [(Yue et al. 2023)](https://aclanthology.org/2023.acl-long.74/) with  1.9M reviews for training,
5000 for validation, and 5000 for testing.
- OpenReview: Crawled and processed ICLR 2023 reviews from [OpenReview website](https://openreview.net/group?id=ICLR.cc/2023/Conference), with  8396 reviews for training, 2798 for validation, and 2798 for testing.
- PubMed: Abstracts of medical papers in [PubMed](https://www.ncbi.nlm.nih.gov/) from 2023/08/01 to 2023/08/07 crawled by [(Yu et al. 2023)](https://openreview.net/forum?id=FKwtKzglFb), with 75316 abstracts for training, 14423 for validation, and 4453 for testng.
