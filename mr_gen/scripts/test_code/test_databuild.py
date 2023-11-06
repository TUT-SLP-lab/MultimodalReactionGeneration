from mr_gen.utils import set_logger
from mr_gen.utils.arg_manager import DataArgmentParser, PreprocessArgmentParser
from mr_gen.databuild import DataBuilder


class AllArgs(DataArgmentParser, PreprocessArgmentParser):
    pass


parser = AllArgs()
args = parser.parse_args()
args.data_dir = "./data/test_site"
args.no_cache_build = True

logger = set_logger("DataBuild-TEST", "./data/test_site/logs/data_build.log")
dbuild = DataBuilder(args, logger)
# dbuild.build()
