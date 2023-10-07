python setup.py install 
python mr_gen/utils/data_analysis/data_alignment.py \
    --target ./data/multimodal_dialogue_clean \
    --output ./data/multimodal_dialogue_formd \
    --procnum 8 \
    --overwrite \
