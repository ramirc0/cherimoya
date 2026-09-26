# cherimoya_cli pipeline command
# Author: Jacob Schreiber <jmschreiber91@gmail.com>


def run(cfg):
	"""Run every step of the pipeline in-process.

	A null `loci`, `negatives` or `model` is produced by an earlier step:
	peaks are called with MACS3, negatives are sampled, and a model is
	trained. The produced value is assigned to the top-level key, so every
	step node that interpolates it sees the new value. Before a step runs,
	its resolved node is saved as `<name>.<command>.yaml`, which reruns
	that step alone with `cherimoya <command> -p <name>.<command>.yaml`.
	With `dry_run`, the step configs are saved and nothing is run. With
	`skip`, nothing is saved or run.


	Parameters
	----------
	cfg: omegaconf.DictConfig
		The composed `pipeline` config. Its inputs must already be resolved
		by `resolve_inputs`, and its top-level keys are overwritten with the
		files the pipeline produces.
	"""

	if cfg.skip:
		return

	import subprocess
	import sys

	import pandas
	from omegaconf import OmegaConf

	from cherimoya.io import normalize_signal_groups

	from . import attribute as attribute_cmd
	from . import fit as fit_cmd
	from . import marginalize as marginalize_cmd
	from . import negatives as negatives_cmd
	from . import seqlets as seqlets_cmd

	preprocess = cfg.preprocessing
	pname = cfg.name

	# The negatives step reads the first peak file, `${loci.0}`, which a
	# bare string does not have.
	for key in ("loci", "negatives"):
		if isinstance(cfg[key], str):
			cfg[key] = [cfg[key]]

	def save(node, command):
		OmegaConf.save(node, "{}.{}.yaml".format(pname, command), resolve=True)

	# Flatten any grouped signals/controls early — every preprocessing
	# step (MACS3 callpeak, bam2bw, file-extension sniffing) operates on
	# the underlying files regardless of how they're grouped for the
	# model. The fit step reads the *original* grouped form from the
	# config, so grouping is preserved end-to-end.
	inputs = OmegaConf.to_container(OmegaConf.masked_copy(cfg,
		["signals", "controls"]))
	signal_files, _ = normalize_signal_groups(inputs["signals"])
	control_files, _ = normalize_signal_groups(inputs["controls"])

	###
	# Step 0.1: Run MACS3 to call peaks if not provided
	###

	if cfg.loci is None:
		if preprocess.verbose:
			print("\nStep 0.1: Call peaks using MACS3.")

		file_format = preprocess.callpeaks_format
		if file_format is None:
			if preprocess.fragments:
				file_format = "FRAG"
			else:
				fname = signal_files[0]

				if fname.endswith(".gz"):
					file_format = fname.split(".")[-2].upper()
				else:
					file_format = fname.split(".")[-1].upper()

				if preprocess.paired_end:
					file_format += "PE"

		cmd_args = [
			"macs3",
			"callpeak",
			"-f",
			file_format,
			"-g",
			str(preprocess.callpeaks_gsize),
			"-n",
			pname,
			"-q",
			str(preprocess.callpeaks_q),
			"-t",
		]

		cmd_args.extend(signal_files)

		if control_files is not None:
			cmd_args += ["-c"]
			cmd_args.extend(control_files)

		if preprocess.fragments:
			cmd_args += ["--max-count", "1"]

		cfg.loci = [pname + "_peaks.narrowPeak"]

		if not cfg.dry_run:
			subprocess.run(cmd_args, check=True)

	###
	# Step 0.2: Convert from SAM/BAMs to bigwigs if provided
	###

	ftypes = ".sam", ".bam", ".bed", ".bed.gz", ".tsv", ".tsv.gz"

	if signal_files[0].endswith(ftypes):
		if preprocess.verbose:
			print("Step 0.2: Convert data to bigWigs")

		cmd_args = [
			"bam2bw",
			"-s",
			cfg.sequences,
			"-n",
			pname,
			"-ps",
			str(preprocess.pos_shift),
			"-ns",
			str(preprocess.neg_shift),
			"-sf",
			str(preprocess.scale_factor),
			"-p",
			"-1",
		]

		if preprocess.read_depth:
			cmd_args += ["-r"]

		if preprocess.unstranded:
			cmd_args += ["-u"]

		if preprocess.fragments:
			cmd_args += ["-f"]

		if preprocess.verbose:
			cmd_args += ["-v"]

		cmd_args += signal_files
		if not cfg.dry_run:
			subprocess.run(cmd_args, check=True)

		# After conversion, rewrite `signals` in the grouped form so the
		# fit step declares strandedness correctly. Unstranded bam2bw
		# produces one bigWig — one unstranded group. Stranded bam2bw
		# produces a (+, -) pair which must be wrapped in an inner list
		# to land as a single stranded group.
		if preprocess.unstranded:
			cfg.signals = [pname + ".bw"]
		else:
			cfg.signals = [[pname + ".+.bw", pname + ".-.bw"]]

	if control_files is not None:
		if control_files[0].endswith(ftypes):
			cmd_args = [
				"bam2bw",
				"-s",
				cfg.sequences,
				"-n",
				pname + ".control",
				"-ps",
				str(preprocess.pos_shift),
				"-ns",
				str(preprocess.neg_shift),
				"-p",
				"-1",
			]

			if preprocess.read_depth:
				cmd_args += ["-r"]

			if preprocess.unstranded:
				cmd_args += ["-u"]

			if preprocess.fragments:
				cmd_args += ["-f"]

			if preprocess.verbose:
				cmd_args += ["-v"]

			cmd_args += control_files
			if not cfg.dry_run:
				subprocess.run(cmd_args, check=True)

			if preprocess.unstranded:
				cfg.controls = [pname + ".control.bw"]
			else:
				cfg.controls = [
					[pname + ".control.+.bw", pname + ".control.-.bw"]
				]

	###
	# Step 0.3: Identify GC-matched negative regions
	###

	if cfg.negatives is None:
		if preprocess.verbose:
			print("\nStep 0.3: Find GC-matched negative regions.")

		save(cfg.negative_sampling, "negatives")
		cfg.negatives = [cfg.negative_sampling.output]

		if not cfg.dry_run:
			negatives_cmd.run(cfg.negative_sampling)

	###
	# Step 1: Fit a Cherimoya model to the provided data
	###

	if cfg.verbose:
		print("\nStep 1: Fitting a Cherimoya model")

	if cfg.model is None:
		cfg.model = pname + ".torch"
		save(cfg.fit, "fit")

		# With more than one device, Lightning starts the other ranks by
		# re-running the current command, which here would re-run the whole
		# pipeline, so the fit gets a command line of its own.
		if not cfg.dry_run and cfg.fit.devices != 1:
			subprocess.run([sys.executable, "-m", "cherimoya_cli", "fit", "-p",
				"{}.fit.yaml".format(pname)], check=True)
		elif not cfg.dry_run:
			fit_cmd.run(cfg.fit)
			# Records the seed fit drew if `random_state` was null.
			save(cfg.fit, "fit")

	###
	# Step 2: Calculate attributions
	###

	if cfg.verbose:
		print("\nStep 2: Calculating attributions")

	save(cfg.attribute, "attribute")
	if not cfg.dry_run:
		attribute_cmd.run(cfg.attribute)

	###
	# Step 3.1: Identify seqlets from attributions
	###

	if cfg.verbose:
		print("\nStep 3.1: Seqlet identification")

	save(cfg.seqlets, "seqlets")
	if not cfg.dry_run:
		seqlets_cmd.run(cfg.seqlets)

	###
	# Step 3.2: Annotate seqlets using motif database
	###

	annotation = cfg.annotation

	if annotation.motifs is not None and not annotation.skip:
		if cfg.verbose:
			print("\nStep 3.2: Seqlet annotation")

		cmd = ["ttl"]
		cmd += ["-f", annotation.sequences]
		cmd += ["-b", annotation.seqlet_filename]
		cmd += ["-s", str(annotation.n_score_bins)]
		cmd += ["-m", str(annotation.n_median_bins)]
		cmd += ["-a", str(annotation.n_target_bins)]
		cmd += ["-c", str(annotation.n_cache)]
		cmd += ["-j", str(annotation.n_jobs)]

		if not annotation.reverse_complement:
			cmd += ["-r"]

		cmd += ["-t", annotation.motifs]

		if not cfg.dry_run:
			with open(annotation.output_filename, "w") as f:
				subprocess.run(cmd, check=True, stdout=f)

			annotated_seqlets = pandas.read_csv(
				annotation.output_filename,
				sep="\t",
				header=None,
				usecols=(3,),
				names=["motifs"],
			)

			seqlet_count = annotated_seqlets.value_counts()
			seqlet_count.to_csv(pname + ".motif_seqlet_count.tsv", sep="\t")

	###
	# Step 4.1: Run TF-MoDISco
	###

	if cfg.verbose:
		print("\nStep 4.1: TF-MoDISco motifs")

	modisco = cfg.modisco_motifs

	cmd = [
		"modisco",
		"motifs",
		"-s",
		cfg.attribute.ohe_filename,
		"-a",
		cfg.attribute.attr_filename,
		"-n",
		str(modisco.n_seqlets),
		"-o",
		modisco.output_filename,
	]

	if modisco.verbose or cfg.verbose:
		cmd += ["-v"]

	if not cfg.dry_run:
		subprocess.run(cmd, check=True)

	###
	# Step 4.2: Generate the tf-modisco report
	###

	report = cfg.modisco_report

	if report.verbose:
		print("\nStep 4.2: TF-MoDISco reports")

	cmd = [
		"modisco",
		"report",
		"-i",
		modisco.output_filename,
		"-o",
		report.output_folder,
		"-s",
		"./",
	]

	if report.motifs is not None:
		cmd += ["-m", report.motifs]

	if not cfg.dry_run:
		subprocess.run(cmd, check=True)

	###
	# Step 5: Marginalization experiments
	###

	if cfg.motifs is None:
		return

	if cfg.verbose:
		print("\nStep 5: Run marginalizations")

	save(cfg.marginalize, "marginalize")
	if not cfg.dry_run:
		marginalize_cmd.run(cfg.marginalize)
