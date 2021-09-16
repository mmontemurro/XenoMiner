include: '../snakemake/conf.sk'

wildcard_constraints:
    assembly = '|'.join([re.escape(x) for x in [ASSEMBLY, ASSEMBLY_M]])

rule all:
    input: 
        expand(DATASET + "/dsk/{k}mer/{assembly}_{chrom}.txt", k=KVALUE, assembly=ASSEMBLY, chrom=CHROMOSOMES),
        expand(DATASET + "/dsk/{k}mer/{assembly}_{chrom}.txt", k=KVALUE, assembly=ASSEMBLY_M, chrom=CHROMOSOMES_M),
        expand(DATASET + "/dsk/{k}mer/{assembly}.txt", k=KVALUE, assembly=[ASSEMBLY, ASSEMBLY_M]),
        #expand(DATASET + "/dsk/{k}mer/" + ASSEMBLY + "_" + ASSEMBLY_M + ".counts", k = KVALUE),
        expand(DATASET + "/dsk/{k}mer/" + ASSEMBLY + "_" + ASSEMBLY_M + ".corr", k = KVALUE),
        expand(DATASET + "/dsk/{k}mer/" + ASSEMBLY + "_" + ASSEMBLY_M + ".corr_filtered", k = KVALUE)

rule kmers_count_chr_dsk:
    input: DATA_DIR+"/{assembly}/chromosomes/{chrom}.fa"
    output: DATASET + "/dsk/{k}mer/{assembly}_{chrom}.h5"
    params: tool=DSK, outdir=DATASET + "/dsk/{k}mer/", min_abundance=1, dsk_log = DATASET + "/dsk/{k}mer/logs/{assembly}_{chrom}.out", cores=DSK_CORES
    log: LOGS_DIR + "/kmers_count_chr_dsk-{assembly}-{chrom}-{k}mer.log"
    benchmark: BENCHMARKS_DIR + "/kmers_count_chr_dsk-{assembly}-{chrom}-{k}mer.tsv"
    shell:
        """
            if [ ! -d {params.outdir} ]; then
                mkdir -p {params.outdir}
            fi 

            if [ ! -d {params.outdir}/logs ]; then
                mkdir -p {params.outdir}/logs
            fi

            {params.tool}  -file {input} -kmer-size {wildcards.k} -abundance-min {params.min_abundance} -nb-cores {params.cores} -out {output} 1> {params.dsk_log} 2> {log}
        """

rule kmers_2ascii_chr:
    input: DATASET + "/dsk/{k}mer/{assembly}_{chrom}.h5"
    output: DATASET + "/dsk/{k}mer/{assembly}_{chrom}.txt"
    params: tool=DSK2ASCII, outdir=DATASET + "/dsk/{k}mer/", dsk_log = DATASET + "/dsk/{k}mer/logs/{assembly}_{chrom}.out"
    log: LOGS_DIR + "/kmers_2ascii_chr-{assembly}-{chrom}-{k}mer.log"
    benchmark: BENCHMARKS_DIR + "/kmers_2ascii_chr-{assembly}-{chrom}-{k}mer.tsv"
    shell:
        """
            {params.tool} -file {input} -out {output} 1>>{params.dsk_log} 2> {log}
        """

rule kmers_count_chr_genome:
    input: DATA_DIR+"/{assembly}/fasta/{assembly}.fa"
    output: DATASET + "/dsk/{k}mer/{assembly}.h5"
    params: tool=DSK, outdir=DATASET + "/dsk/{k}mer", min_abundance=1, dsk_log = DATASET + "/dsk/{k}mer/logs/{assembly}.out", cores=DSK_CORES
    log: LOGS_DIR + "/kmers_count_chr_genome-{assembly}-{k}mer.log"
    benchmark: BENCHMARKS_DIR + "/kmers_count_chr_genome-{assembly}-{k}mer.tsv"
    shell:
        """
            if [ ! -d {params.outdir} ]; then
                mkdir -p {params.outdir}
            fi 

            if [ ! -d {params.outdir}/logs ]; then
                mkdir -p {params.outdir}/logs
            fi

            {params.tool}  -file {input} -kmer-size {wildcards.k} -abundance-min {params.min_abundance} -nb-cores {params.cores} -out {output} 1> {params.dsk_log} 2> {log}
        """

rule kmers_2ascii_genome:
    input: DATASET + "/dsk/{k}mer/{assembly}.h5"
    output: DATASET + "/dsk/{k}mer/{assembly}.txt"
    params: tool=DSK2ASCII, outdir=DATASET + "/dsk/{k}mer", dsk_log = DATASET + "/dsk/{k}mer/logs/{assembly}.out"
    log: LOGS_DIR + "/kmers_2ascii_genome-{assembly}-{k}mer.log"
    benchmark: BENCHMARKS_DIR + "/kmers_2ascii_genome-{assembly}-{k}mer.tsv"
    shell:
        """
            {params.tool} -file {input} -out {output} 1>> {params.dsk_log} 2> {log}
        """

#join and intersect kmer lists
rule kmers_stats:
    input: expand(DATASET + "/dsk/{k}mer/{assembly}.txt", k=KVALUE, assembly=[ASSEMBLY, ASSEMBLY_M])
    output:
          DATASET + "/dsk/{k}mer/" + ASSEMBLY + "_" + ASSEMBLY_M + ".counts"
    params:
        outdir= DATASET + "/dsk/{k}mer/"
    shell:
        """
            assembly1=$(basename {input[0]} .txt)
            assembly2=$(basename {input[1]} .txt)
            echo -e "kmer\t${{assembly1}}\t${{assembly2}}" > {output}; join -e 0 -o auto -a 1 -a 2  <(sort {input[0]}) <(sort {input[1]}) | awk -v OFS="\t" '$1=$1' >> {output}
        """

#normalize kmer counts
rule kmers_freq:
    input: DATASET + "/dsk/{k}mer/" + ASSEMBLY + "_" + ASSEMBLY_M + ".counts"
    output: DATASET + "/dsk/{k}mer/" + ASSEMBLY + "_" + ASSEMBLY_M + ".freqs"
    run:
        import pandas as pd 
        df = pd.read_csv(input[0], sep="\t", index_col=0)
        df_stats = df / df.sum()
        df.to_csv(output[0], sep="\t")

rule kmers_diff:
    input: DATASET + "/dsk/{k}mer/" + ASSEMBLY + "_" + ASSEMBLY_M + ".freqs"
    output: DATASET + "/dsk/{k}mer/" + ASSEMBLY + "_" + ASSEMBLY_M + ".diff",
            DATASET + "/dsk/{k}mer/" + ASSEMBLY + "_" + ASSEMBLY_M + ".diff_abs",
            DATASET + "/dsk/{k}mer/" + ASSEMBLY + "_" + ASSEMBLY_M + ".freqs_filtered"
    params: assembly1 = ASSEMBLY, assembly2 = ASSEMBLY_M, tool=DSKSTATS, outprefix=DATASET + "/dsk/{k}mer/" + ASSEMBLY + "_" + ASSEMBLY_M 
    shell:
        """
            python {params.tool} -i {input} -o {params.outprefix} -a1 {params.assembly1} -a2 {params.assembly2}
        """

# compute pearson correlation between the distributions
rule pearsonr:
    input: DATASET + "/dsk/{k}mer/" + ASSEMBLY + "_" + ASSEMBLY_M + ".freqs"
    output: DATASET + "/dsk/{k}mer/" + ASSEMBLY + "_" + ASSEMBLY_M + ".corr"
    params: assembly1 = ASSEMBLY, assembly2 = ASSEMBLY_M
    run:
        from scipy.stats import pearsonr
        import pandas as pd 
        df = pd.read_csv(input[0], sep="\t", index_col=0)
        s, p = pearsonr(df[params.assembly1], df[params.assembly2])
        with open(output[0], "w" ) as f:
            f.write("pearson_corr_coeff\tpvalue\n{}\t{}\n".format(s, p))

rule pearsonr_filtered:
    input: DATASET + "/dsk/{k}mer/" + ASSEMBLY + "_" + ASSEMBLY_M + ".freqs_filtered"
    output: DATASET + "/dsk/{k}mer/" + ASSEMBLY + "_" + ASSEMBLY_M + ".corr_filtered"
    params: assembly1 = ASSEMBLY, assembly2 = ASSEMBLY_M
    run:
        from scipy.stats import pearsonr
        import pandas as pd 
        df = pd.read_csv(input[0], sep="\t", index_col=0)
        s, p = pearsonr(df[params.assembly1], df[params.assembly2])
        with open(output[0], "w" ) as f:
            f.write("pearson_corr_coeff\tpvalue\n{}\t{}\n".format(s, p))
    

