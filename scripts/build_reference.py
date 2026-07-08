"""Build the HCP normative reference by running every HCP subject through the
same extraction path. Fill in subject discovery + label loading once the input
spec and atlas are frozen.
"""

from __future__ import annotations

from sftoolbox.config import DEFAULT
from sftoolbox import io, pipeline, reference


def main():
    cfg = DEFAULT
    # TODO: discover HCP subject dirs and the atlas label volume.
    subject_dirs: dict[str, str] = {}          # {subject_id: path}
    labels_3d = None                            # io.load_nifti_data(atlas_path)

    values, ids = [], []
    for sid, root in subject_dirs.items():
        subj = io.Subject.from_dir(sid, root)
        values.append(pipeline.extract_subject(subj, labels_3d, cfg))
        ids.append(sid)

    reference.build_reference(values, ids, cfg)
    print(f"Wrote reference for {len(ids)} subjects -> {cfg.reference_path}")


if __name__ == "__main__":
    main()
