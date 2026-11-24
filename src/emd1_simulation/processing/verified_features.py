"""Verify the published NEDD4L interval against a checksummed hg38-to-hg19
assembly chain and the first exon of the cited RefSeq transcript.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import re

NEDD4L_TRANSCRIPT = "NM_001144966.3"
NEDD4L_PUBLICATION = "10.1038/s41467-021-22469-6"
NEDD4L_PRINTED_INTERVAL = "NC_000018.10:58221610-58221760"
QUARANTINED_NEDD4L_INTERVAL = ("chr18", 55712458, 55712608)


def _text_lines(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    return opener(path, "rt", encoding="utf-8")


def _evidence_file(record: dict, key: str, base: Path) -> Path:
    item = record.get(key)
    if not isinstance(item, dict):
        raise ValueError(f"Missing {key} evidence record")
    if not str(item.get("source_url", "")).startswith("https://"):
        raise ValueError(f"{key} requires its source URL")
    expected = str(item.get("sha256", ""))
    if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError(f"{key} requires a recorded SHA-256")
    evidence = Path(str(item.get("path", "")))
    if not evidence.is_absolute():
        evidence = base / evidence
    if not evidence.is_file():
        raise FileNotFoundError(f"Missing {key} evidence file: {evidence}")
    digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(f"{key} evidence checksum mismatch: {evidence}")
    return evidence


def _interval(value: object, name: str) -> tuple[str, int, int]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain chrom, start and end")
    chrom, start, end = value.get("chrom"), value.get("start"), value.get("end")
    if (not isinstance(chrom, str) or not chrom.startswith("chr")
            or type(start) is not int or type(end) is not int
            or start < 0 or end <= start):
        raise ValueError(f"Invalid zero-based half-open {name}")
    return chrom, start, end


def chain_mapping(chain: Path, source: tuple[str, int, int]) -> tuple[str, int, int, str]:
    """Map a complete interval in one block; reject ambiguity and split maps.

    UCSC ``hg38ToHg19`` chains use hg38 as t and hg19 as q. The UCSC t strand
    is '+'. q coordinates are converted from strand coordinates when needed.
    This deliberately does not join blocks across indels or gaps.
    """
    chrom, start, end = source
    matches = set()
    header = None
    chain_matches = set()
    t_pos = q_pos = 0
    t_end = q_end = 0
    with _text_lines(chain) as stream:
        for line in stream:
            fields = line.split()
            if not fields:
                if header is not None:
                    raise ValueError("Unfinished UCSC chain before a blank line")
                continue
            if fields[0] == "chain":
                if header is not None:
                    raise ValueError("Unfinished UCSC chain before a new header")
                if len(fields) != 13:
                    raise ValueError("Malformed UCSC chain header")
                score, t_size, q_size = int(fields[1]), int(fields[3]), int(fields[8])
                t_start, t_end = int(fields[5]), int(fields[6])
                q_start, q_end = int(fields[10]), int(fields[11])
                chain_id = int(fields[12])
                if (score < 0 or chain_id < 0 or t_size <= 0 or q_size <= 0
                        or not 0 <= t_start < t_end <= t_size
                        or not 0 <= q_start < q_end <= q_size
                        or fields[4] != "+" or fields[9] not in {"+", "-"}):
                    raise ValueError("Invalid UCSC chain bounds, sizes, score, identifier or strand")
                header = fields
                chain_matches = set()
                t_pos, q_pos = t_start, q_start
                continue
            if header is None:
                raise ValueError("Chain block without a header")
            if len(fields) not in (1, 3):
                raise ValueError("Malformed UCSC chain block")
            size = int(fields[0])
            if size <= 0:
                raise ValueError("Nonpositive UCSC chain block length")
            block_t_end, block_q_end = t_pos + size, q_pos + size
            if block_t_end > t_end or block_q_end > q_end:
                raise ValueError("UCSC chain block exceeds its declared header bounds")
            if header[2] == chrom and t_pos <= start and end <= t_pos + size:
                q_start = q_pos + start - t_pos
                mapped_q_end = q_pos + end - t_pos
                q_strand = header[9]
                if q_strand == "-":
                    q_size = int(header[8])
                    q_start, mapped_q_end = q_size - mapped_q_end, q_size - q_start
                chain_matches.add((header[7], q_start, mapped_q_end, q_strand))
            if len(fields) == 3:
                dt, dq = int(fields[1]), int(fields[2])
                if dt < 0 or dq < 0:
                    raise ValueError("Negative UCSC chain gap")
                t_pos, q_pos = block_t_end + dt, block_q_end + dq
                if t_pos >= t_end or q_pos >= q_end:
                    raise ValueError("Nonterminal UCSC chain block leaves no room for its terminal block")
            else:
                if block_t_end != t_end or block_q_end != q_end:
                    raise ValueError("Terminal UCSC chain block does not reach the declared header ends")
                matches.update(chain_matches)
                header = None
    if header is not None:
        raise ValueError("Unfinished UCSC chain at end of file; terminal block required")
    if len(matches) != 1:
        raise ValueError(
            "Source interval must have one complete, unique single-block chain mapping; "
            f"found {len(matches)}. Resolve split/ambiguous mappings explicitly.")
    return next(iter(matches))


def _in_cited_first_exon(annotation: Path, interval: tuple[str, int, int],
                         transcript: str, strand: str) -> None:
    chrom, start, end = interval
    accession = transcript.split(".")[0]
    exons = set()
    with _text_lines(annotation) as stream:
        for line in stream:
            f = line.rstrip("\n").split("\t")
            if len(f) < 13 or f[1].split(".")[0] != accession or f[12] != "NEDD4L":
                continue
            if "." in f[1] and f[1] != transcript:
                continue
            if f[2] != chrom or f[3] != strand:
                continue
            starts = [int(x) for x in f[9].rstrip(",").split(",")]
            ends = [int(x) for x in f[10].rstrip(",").split(",")]
            if len(starts) != len(ends) or not starts:
                raise ValueError("Invalid cited transcript annotation")
            i = 0 if strand == "+" else -1
            exons.add((starts[i], ends[i]))
    if len(exons) != 1:
        raise ValueError("Target annotation must identify one first exon of NM_001144966")
    exon_start, exon_end = next(iter(exons))
    if not exon_start <= start < end <= exon_end:
        raise ValueError("Mapped interval is outside the cited transcript's first exon")


def load_verified_feature(path: str | Path, gene: str = "NEDD4L") -> dict:
    """Check source identity, file hashes, whole-interval mapping and first-exon containment.

    RefGene annotations may omit version suffixes. Record the full published
    accession and hash the annotation used for verification.
    """
    path = Path(path).resolve()
    r = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version": 1, "gene": gene,
        "transcript": NEDD4L_TRANSCRIPT,
        "publication_doi": NEDD4L_PUBLICATION,
        "published_interval": NEDD4L_PRINTED_INTERVAL,
        "source_assembly": "hg38", "target_assembly": "hg19",
        "coordinate_convention": "0-based-half-open",
        "verification_method": "ucsc_chain_and_refseq_first_exon",
    }
    if gene != "NEDD4L" or any(r.get(k) != v for k, v in required.items()):
        raise ValueError("Feature record does not match the cited NEDD4L source/assemblies")
    interpretation = r.get("published_coordinate_interpretation")
    if interpretation not in {"printed_interval_is_0_based_half_open", "printed_interval_is_1_based_closed"}:
        raise ValueError("Explicitly declare the interpretation of the published coordinate convention")
    if not isinstance(r.get("coordinate_interpretation_evidence"), str) or not r["coordinate_interpretation_evidence"].strip():
        raise ValueError("Record the evidence supporting the publication-coordinate interpretation")
    source = _interval(r.get("source_interval"), "source_interval")
    target = _interval(r.get("target_interval"), "target_interval")
    expected_start = 58221610 if interpretation == "printed_interval_is_0_based_half_open" else 58221609
    if source != ("chr18", expected_start, 58221760):
        raise ValueError("Source interval disagrees with the declared publication-coordinate interpretation")
    if target == QUARANTINED_NEDD4L_INTERVAL:
        raise ValueError("The historical unsupported NEDD4L interval is quarantined")
    chain = _evidence_file(r, "assembly_chain", path.parent)
    annotation = _evidence_file(r, "target_refgene", path.parent)
    mapped = chain_mapping(chain, source)
    if mapped[:3] != target:
        raise ValueError("Declared target interval does not match the checksummed assembly chain")
    if mapped[3] != "+":
        raise ValueError("Cited NEDD4L transcript requires a forward-strand target mapping")
    _in_cited_first_exon(annotation, target, NEDD4L_TRANSCRIPT, mapped[3])
    return {
        "gene": gene, "chrom": target[0], "start_0based": target[1], "end_0based": target[2],
        "transcript": NEDD4L_TRANSCRIPT, "source": NEDD4L_PUBLICATION + " Fig. 3f",
        "status": "mapping_verified_against_supplied_evidence",
        "feature_record_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "assembly_chain_sha256": r["assembly_chain"]["sha256"],
        "target_refgene_sha256": r["target_refgene"]["sha256"],
        "published_coordinate_interpretation": interpretation,
        "coordinate_interpretation_evidence": r["coordinate_interpretation_evidence"],
    }
