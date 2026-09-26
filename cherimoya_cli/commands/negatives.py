# cherimoya_cli negatives command
# Author: Jacob Schreiber <jmschreiber91@gmail.com>


def run(cfg):
	"""Sample GC-matched negatives for a peak file and write them as a BED
	file.


	Parameters
	----------
	cfg: omegaconf.DictConfig
		A config typed by `cherimoya_cli.config.NegativesConfig`.
	"""

	from tangermeme.match import extract_matching_loci

	matched_loci = extract_matching_loci(
		loci=cfg.peaks,
		fasta=cfg.fasta,
		gc_bin_width=cfg.bin_width,
		max_n_perc=cfg.max_n_perc,
		bigwig=cfg.bigwig,
		signal_beta=cfg.beta,
		in_window=cfg.in_window,
		out_window=cfg.out_window,
		chroms=None,
		verbose=cfg.verbose,
		n_jobs=1,
	)

	matched_loci.to_csv(cfg.output, header=False, sep="\t", index=False)
