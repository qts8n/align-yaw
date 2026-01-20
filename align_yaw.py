import logging
import sys

import cv2

from equirect_shift import AlignConfig, AlignmentError, align_panoramas
from equirect_shift.cli import build_parser

logger = logging.getLogger(__name__)


def main(argv=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
        stream=sys.stdout)

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = AlignConfig.from_namespace(args)
    except ValueError as exc:
        raise SystemExit(str(exc))

    try:
        result = align_panoramas(config)
    except AlignmentError as exc:
        raise SystemExit(str(exc))

    ok = cv2.imwrite(config.out, result.aligned)
    logger.info("Wrote: %s (%s)", config.out, "ok" if ok else "FAILED")

    if config.save_mask:
        ok_mask = cv2.imwrite(config.save_mask, result.mask_full)
        logger.info("Saved mask: %s (%s)", config.save_mask, "ok" if ok_mask else "FAILED")


if __name__ == "__main__":
    main()
