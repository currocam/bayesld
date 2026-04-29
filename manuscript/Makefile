.PHONY: pipeline clean

pipeline: flake.lock
	nix develop --command nextflow run main.nf -resume -qs 1

clean:
	rm -rf work .nextflow .nextflow.log*
