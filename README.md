# PubMed Vectors

in this repository, i demonstrate how one can download a whole set of abstracts from PubMed (`download_pubmed.py`) into a sqlite3 database file, embed each abstract (title if the abstract is empty) into a high-dimensional dense vector using Nomic's embedding model (`embed_pubmed_st.py`) and search through the abstracts given a new query (`test_pubmed.py`). it is the most naive version of such a dense vector based retrieval system, and obviously, is very very slow (expect about 5 minutes for a single query, unless you're loading up the whole .h5 file on memory.)

due to my stupidity earlier, `embed_pubmed_st.py` uses numpy memmap'd arrayes, which is absolutely unnecessary and wasteful. it is thus necessary to convert these arrays into a hdf5 file with `convert_dat_to_h5.py`. mea culpa. 

here are some samples:

### What is the role of GLP-1 and GLP-1 agonists in losing excess weight?

1. Rank 1, Similarity: 0.862619161605835
PMID: 37445623
Title: Emerging Role of GLP-1 Agonists in Obesity: A Comprehensive Review of Randomised Controlled Trials.
Authors: Mihaela-Simona Popoviciu, Lorena Păduraru, Galal Yahya, Kamel Metwally, Simona Cavalu

2. Rank 2, Similarity: 0.8599241375923157
PMID: 25202976
Title: Glucagon-like peptide-1 receptors in the brain: controlling food intake and body weight.
Authors: Laurie L Baggio, Daniel J Drucker

3. Rank 3, Similarity: 0.8542712330818176
PMID: 37460364
Title: Are GLP-1R agonists the long-sought-after panacea for obesity?
Authors: Georgina K C Dowsett, Giles S H Yeo

4. Rank 4, Similarity: 0.8532418012619019
PMID: 26961053
Title: [GLP-1 agonist supports weight loss].
Authors: Maria Weiß

5. Rank 5, Similarity: 0.8530821204185486
PMID: 36321278
Title: Weight loss between glucagon-like peptide-1 receptor agonists and bariatric surgery in adults with obesity: A systematic review and meta-analysis.
Authors: Shohinee Sarma, Patricia Palcu

### What are the biologies of TEAD?

1. Rank 1, Similarity: 0.7603017687797546
PMID: 36063664
Title: A chemical perspective on the modulation of TEAD transcriptional activities: Recent progress, challenges, and opportunities.
Authors: Jianfeng Lou, Yuhang Lu, Jing Cheng, Feilong Zhou, Ziqin Yan, Daizhou Zhang, Xiangjing Meng, Yujun Zhao

2. Rank 2, Similarity: 0.759563684463501
PMID: 35077390
Title: Integrative bioinformatics and experimental analysis revealed TEAD as novel prognostic target for hepatocellular carcinoma and its roles in ferroptosis regulation.
Authors: Xinxin Ren, Xiang Wang, Yuanliang Yan, Xi Chen, Yuan Cai, Qiuju Liang, Bi Peng, Zhijie Xu, Qingchun He, Fanhua Kang, Jianbo Li, Wenqin Zhang, Qianhui Hong, Jinwu Peng, Muzhang Xiao

3. Rank 3, Similarity: 0.7527551651000977
PMID: 27421669
Title: An evolutionary, structural and functional overview of the mammalian TEAD1 and TEAD2 transcription factors.
Authors: André Landin-Malt, Ataaillah Benhaddou, Alain Zider, Domenico Flagiello

4. Rank 4, Similarity: 0.7498442530632019
PMID: 34729310
Title: Discovery of a subtype-selective, covalent inhibitor against palmitoylation pocket of TEAD3.
Authors: Tian Lu, Yong Li, Wenchao Lu, Twgm Spitters, Xueyu Fang, Jun Wang, Simian Cai, Jing Gao, Yanting Zhou, Zhe Duan, Huan Xiong, Liping Liu, Qi Li, Hualiang Jiang, Kaixian Chen, Hu Zhou, Hua Lin, Huijin Feng, Bing Zhou, Christopher L Antos, Cheng Luo

5. Rank 5, Similarity: 0.7495083212852478
PMID: 33317954
Title: Association of subcellular localization of TEAD transcription factors with outcome and progression in pancreatic ductal adenocarcinoma.
Authors: Richard Drexler, Rebecca Fahy, Mirco Küchler, Kim C Wagner, Tim Reese, Mareike Ehmke, Bernd Feyerabend, Moritz Kleine, Karl J Oldhafer

