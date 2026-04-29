.PHONY: pipeline pipeline-hpc clean

pipeline:
	nextflow run main.nf -resume -profile local

pipeline-hpc:
	nextflow run main.nf -resume -profile hpc -work-dir work

clean:
	rm -rf work .nextflow .nextflow.log*
