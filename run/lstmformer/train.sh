NAME=cradle-01

python setup.py install
python mr_gen/model/lstmformer/trainer.py \
    --config-path=./ \
    --config-name=config.yaml \
    name="$NAME" \
    no_cache_build=false \
    clear_cache=false \
