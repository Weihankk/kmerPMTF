# kmerPMTF

# Usage

1. Download dataset from here: https://figshare.com/articles/dataset/kmerPMTF-data/24916023
2. Clone kmerPMTF into your platform:
   
   ```git clone https://github.com/Weihankk/kmerPMTF.git```

3. Move dataset to kmerPMTF directory and decompose it:

   ```
   mv data.tar.gz /your/dir/kmerPMTF
   cd /ypur/dir/kmerPMTF
   tar zxvf data.tar.gz
   ```

4. Running step 1, process dataset. (Take Arabidopsis thaliana as an example)

   ```
   python -u step1_graphconstruct.py --prefix Arabidopsis_thaliana --miRNA_fa ./data/Arabidopsis_thaliana/PmiREN/Arabidopsis_thaliana_mature.fa --target ./data/Arabidopsis_thaliana/PmiREN/Arabidopsis_thaliana_targetGene.txt --transcript ./data/Arabidopsis_thaliana/Genome/Athaliana_167_TAIR10.transcript_primaryTranscriptOnly.fa --kmer 7 > Arabidopsis_thaliana_step1.log &
   ```
5. Running step 2, encoded.

   ```
   python -u step2_encoded.py --prefix Arabidopsis_thaliana --D 256 > Arabidopsis_thaliana_D256_step2.log
   ```

6. Training

   ```
   python step3_train.py --method ChebConv --prefix Prunus_persica --device cuda:0 --lr 0.001 --epoch 3000 --kfold 10 --chebconvk 2 --D 256 --sim cosine
   ```

Result are formatted and printed on screen like follow:
<img width="818" alt="image" src="https://github.com/Weihankk/kmerPMTF/assets/33243134/c12d0c16-5e9c-4f80-b8dd-510cb7924767">
