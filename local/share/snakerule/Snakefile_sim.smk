include: '../snakemake/conf.sk'

rule all:
    input: #expand(DATASET + "/sim_reads/{assembly}_chr1_sim_R{n}.fq", assembly=[ASSEMBLY, ASSEMBLY_M], n=[1,2]),
            DATASET + "/sim_reads/dataset/chr1_sim.dat",
            DATASET + "/sim_reads/dataset/chr1_sim.lab",
            DATASET + "cnn-chr1_sim_1epoch.done"

# simulation of paired-end reads of 150bp  
#       - with the mean fragment size 500 and standard deviation 10
#       - with fold coverage 1X
#       - using HiSeq 2500 sequencing system
#       - using the built-in combined read quality profiles
#       - with a fixed seed (777) (otherwise, segmentation fault)
#       - no aln files
rule art_illumina:
    input: lambda wildcards: CHR1[wildcards.assembly]
    output: DATASET + "/sim_reads/fastq/{assembly}_chr1_sim_R1.fq", DATASET + "/sim_reads/fastq/{assembly}_chr1_sim_R2.fq"
    params: outdir = DATASET + "/sim_reads/", 
            outprefix = DATASET + "/sim_reads/fastq/{assembly}_chr1_sim_R",
            tool = ART, 
            seed = 777,
            read_len = 150,
            cov = 1,
            mean_frag_size = 500,
            st_dev = 10,
            seq_sys = "HS25"
    shell:
        """
            if [ ! -d {params.outdir} ]; then
                mkdir -p {params.outdir}
            fi

            {params.tool} -ss {params.seq_sys} -i {input} -rs {params.seed} -o {params.outprefix} -l {params.read_len} -f {params.cov} -p -m {params.mean_frag_size} -s {params.st_dev} -na
        """

rule kmers_frequency:
    input:  fq = DATASET + "/sim_reads/fastq/{assembly}_chr1_sim_R{n}.fq"
    output: DATASET + "/sim_reads/kmer_freqs/{assembly}_chr1_sim_R{n}.freqs"
    log: LOGS_DIR + "/kmers_frequency-{assembly}_chr1_sim_R{n}.log"
    benchmark: BENCHMARKS_DIR + "/kmers_frequency-{assembly}_chr1_sim_R{n}.tsv"
    params: tool=FASTA2MATRIX, k=12, outdir=DATASET + "/sim_reads/kmer_freqs/", kmers_list = KMERS_DICT, file_type="fastq"
    shell:
        """
        if [ ! -d {params.outdir} ]; then
            mkdir -p {params.outdir}
        fi

        python {params.tool} -k {params.k} -i {input.fq} -d {params.kmers_list} -t {params.file_type} -o {output} -g -f 2> {log}
        """

rule join_scaffolds:
    input: expand(DATASET + "/sim_reads/kmer_freqs/{{assembly}}_chr1_sim_R{n}.freqs", n=[1, 2])
    output: DATASET + "/sim_reads/kmer_freqs/{assembly}_chr1_sim_merged.freqs"
    log: LOGS_DIR + "/join_scaffolds-{assembly}_chr1_sim.log"
    benchmark: BENCHMARKS_DIR + "/join_scaffolds-{assembly}_chr1_sim.tsv"
    params: tool=JOIN_SCAFFOLDS
    shell:
        """
            python {params.tool} -R1 {input[0]} -R2 {input[1]} -o {output} 2> {log}
        """

rule label:
    input: h=expand(DATASET + "/sim_reads/kmer_freqs/{assembly}_chr1_sim_merged.freqs", assembly=ASSEMBLY),
            m=expand(DATASET + "/sim_reads/kmer_freqs/{assembly}_chr1_sim_merged.freqs", assembly=ASSEMBLY_M)
    output: DATASET + "/sim_reads/dataset/chr1_sim.dat", DATASET + "/sim_reads/dataset/chr1_sim.lab"
    params: tool=LABELLING,  outdir=DATASET + "/sim_reads/dataset", outprefix=DATASET + "/sim_reads/dataset/chr1_sim"
    shell:
        """
            if [ ! -d {params.outdir} ]; then
                mkdir -p {params.outdir}
            fi

            python {params.tool} -hu {input.h} -m {input.m} -o {params.outprefix}
        """

rule cnn:
    input: DATASET + "/sim_reads/dataset/chr1_sim.dat", DATASET + "/sim_reads/dataset/chr1_sim.lab"
    output: touch(DATASET + "cnn-chr1_sim_1epoch.done")
    params: in_prefix = DATASET + "/sim_reads/dataset/chr1_sim", out_prefix = DATASET + "/sim_reads/results/chr1_sim_1epoch", 
            outdir = DATASET + "/sim_reads/results", threads = 20, cnn = CNN
    shell:
        """
            if [ ! -d {params.outdir} ]; then
                mkdir -p {params.outdir}
            fi

            python {params.cnn} -i {params.in_prefix} -o {params.out_prefix} -t {params.threads}
        """
