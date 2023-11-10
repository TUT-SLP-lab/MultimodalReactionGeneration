python setup.py install
python mr_gen/utils/visualize/model_visualize.py \
    --config-path=/home/mikawa/lab/MultimodalReactionGeneration2/mr_gen/model/simple_lstm \
    --config-name=best_conf.yaml \
    model_type="simple_lstm" \
    model_path="log/simple_lstm/Multimodal-Head-Motion-Prediction/DEV15/checkpoints/model.ckpt" \
    model_conf="/home/mikawa/lab/MultimodalReactionGeneration2/mr_gen/model/simple_lstm/best_conf.yaml" \
    movie_path="/home/mikawa/lab/MultimodalReactionGeneration/data/unprocess/multimodal_dialogue_formd/data004/comp.mp4" \
    audio_path="/home/mikawa/lab/MultimodalReactionGeneration/data/unprocess/multimodal_dialogue_formd/data004/comp.wav" \
    output_path="data/visualize/simple_lstm/data004_comp.mp4" \

# audio_path の置いてあるディレクトリに必ず相手音声をおくこと