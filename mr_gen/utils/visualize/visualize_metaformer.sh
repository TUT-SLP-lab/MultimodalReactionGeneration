python setup.py install
python mr_gen/utils/visualize/model_visualize.py \
    --config-path=/home/mikawa/lab/MultimodalReactionGeneration2/outputs/2023-11-11/15-06-47/.hydra \
    --config-name=config.yaml \
    model_type="simple_lstm" \
    model_path="log/simple_lstm/Multimodal-Head-Motion-Prediction/cradle-02B/checkpoints/model.ckpt" \
    movie_path="/home/mikawa/lab/MultimodalReactionGeneration/data/unprocess/multimodal_dialogue_formd/data004/comp.mp4" \
    audio_path="/home/mikawa/lab/MultimodalReactionGeneration/data/unprocess/multimodal_dialogue_formd/data004/comp.wav" \
    output_path="data/visualize/simple_lstm/data004_comp_02B.mp4" \

# audio_path の置いてあるディレクトリに必ず相手音声をおくこと