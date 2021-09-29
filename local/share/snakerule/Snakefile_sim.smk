include: '../snakemake/conf.sk'

rule all:
    input: expand(DATASET + "/sim_reads/{assembly}_chr1_sim_R{n}.fq", assembly=[ASSEMBLY, ASSEMBLY_M], n=[1,2]),

# simulation of paired-end reads of 150bp  
#       - with the mean fragment size 500 and standard deviation 10
#       - with fold coverage 1X
#       - using HiSeq 2500 sequencing system
#       - using the built-in combined read quality profiles
#       - with a fixed seed (777) (otherwise, segmentation fault)
#       - no aln files
rule art_illumina:
    input: lambda wildcards: CHR1[wildcards.assembly]
    output: DATASET + "/sim_reads/{assembly}_chr1_sim_R1.fq", DATASET + "/sim_reads/{assembly}_chr1_sim_R2.fq"
    params: outdir = DATASET + "/sim_reads/", 
            outprefix = DATASET + "/sim_reads/{assembly}_chr1_sim_R",
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
            