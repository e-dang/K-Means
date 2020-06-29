
SERIAL = 'serial'
OMP = 'omp'
MPI = 'mpi'
HYBRID = 'hybrid'


MEMORY = 10
HOURS = 2
MINUTES = 0
SECONDS = 0


class JobWriter:
    def __init__(self, parallelism, memory, hours, minutes, seconds, coreset=False, **kwargs):
        self.parallelism = parallelism
        self.memory = memory
        self.hours = hours
        self.minutes = minutes
        self.seconds = seconds
        self.num_threads = kwargs.get('num_threads', None)
        self.num_procs = kwargs.get('num_procs', None)
        self.coreset = coreset
        self.job = ''
        self.coreset_str = 'coreset' if self.coreset else ''

    def write(self, filepath):
        self.__write_header()
        self.__write_module_list()
        self.__write_misc()
        self.__write_programs()

        with open(filepath, 'w') as file:
            file.write(self.job)

    def __write_header(self):
        self.job += '#!/bin/bash\n'
        self.job += '#$ -cwd\n'
        self.job += '#$ -j y\n'

        if self.parallelism == SERIAL:
            if not self.coreset:
                self.job += f'#$ -o {SERIAL}.txt\n'
            else:
                self.job += f'#$ -o {SERIAL}_coreset.txt\n'
            self.job += f'#$ -l h_data={self.memory}G,h_rt={self.hours:02d}:{self.minutes:02d}:{self.seconds:02d},exclusive\n'
        elif self.parallelism == OMP:
            mem_per_thread = self.memory / self.num_threads
            self.job += f'#$ -o {OMP}_{self.coreset_str}_{self.num_threads}.txt\n'
            self.job += f'#$ -l h_data={mem_per_thread}G,h_rt={self.hours:02d}:{self.minutes:02d}:{self.seconds:02d},h_vmem={self.memory}G,exclusive\n'
            self.job += f'#$ -pe shared {self.num_threads}\n'
        else:
            mem_per_proc = self.memory / self.num_procs
            if self.parallelism == MPI:
                self.job += f'#$ -o {MPI}_{self.coreset_str}_{self.num_procs}.txt\n'
                parallel_env = f'#$ -pe dc* {self.num_procs}\n'
            else:
                if not self.coreset:
                    self.job += f'#$ -o {HYBRID}_{self.num_procs}.txt\n'
                else:
                    self.job += f'#$ -o {HYBRID}_coreset_{self.num_procs}.txt\n'
                parallel_env = f'#$ -pe node* {self.num_procs}\n'
            self.job += f'#$ -l h_data={mem_per_proc}G,h_rt={self.hours:02d}:{self.minutes:02d}:{self.seconds:02d},h_vmem={self.memory}G,exclusive\n'
            self.job += parallel_env

        self.job += '\n'

    def __write_module_list(self):
        self.job += '. /u/local/Modules/default/init/modules.sh\n'
        self.job += 'module load gcc/7.2.0\n'
        self.job += 'module load boost/1_71_0\n'
        self.job += 'module load openmpi/3.0.0\n\n'

    def __write_misc(self):
        if self.parallelism == OMP:
            self.job += f'export OMP_NUM_THREADS={self.num_threads}\n\n'
            if not self.coreset:
                self.job += 'echo "num threads = ${OMP_NUM_THREADS}" > ' + \
                    f'{self.parallelism}_lloyd_{self.num_threads}.txt\n\n'
            self.job += 'echo "num threads = ${OMP_NUM_THREADS}" > ' + \
                f'{self.parallelism}_{self.coreset_str}_optlloyd_{self.num_threads}.txt\n\n'
        elif self.parallelism == HYBRID:
            self.job += f'export OMP_NUM_THREADS=$(cat /proc/cpuinfo | grep ^processor | wc -l )\n'
            if not self.coreset:
                self.job += 'echo "num threads = ${OMP_NUM_THREADS}" > ' + \
                    f'{self.parallelism}_lloyd_{self.num_procs}.txt\n\n'
            self.job += 'echo "num threads = ${OMP_NUM_THREADS}" > ' + \
                f'{self.parallelism}_{self.coreset_str}_optlloyd_{self.num_procs}.txt\n\n'

    def __write_programs(self):
        if self.parallelism == SERIAL and not self.coreset:
            self.job += f'../build/kmeans_serial_lloyd > {self.parallelism}_lloyd.txt\n'
            self.job += f'../build/kmeans_serial_optlloyd > {self.parallelism}_optlloyd.txt\n'
        if self.parallelism == SERIAL and self.coreset:
            self.job += f'../build/kmeans_serial_coreset_optlloyd > {self.parallelism}_coreset_optlloyd.txt\n'
        elif self.parallelism == OMP and not self.coreset:
            self.job += f'../build/kmeans_omp_lloyd >> {self.parallelism}_lloyd_{self.num_threads}.txt\n'
            self.job += f'../build/kmeans_omp_optlloyd >> {self.parallelism}_optlloyd_{self.num_threads}.txt\n'
        elif self.parallelism == OMP and self.coreset:
            self.job += f'../build/kmeans_omp_coreset_optlloyd >> {self.parallelism}_coreset_optlloyd_{self.num_threads}.txt\n'
        elif self.parallelism == MPI and not self.coreset:
            self.job += f'mpirun ../build/kmeans_mpi_lloyd > {self.parallelism}_lloyd_{self.num_procs}.txt\n'
            self.job += f'mpirun ../build/kmeans_mpi_optlloyd  > {self.parallelism}_optlloyd_{self.num_procs}.txt\n'
        elif self.parallelism == MPI and self.coreset:
            self.job += f'mpirun ../build/kmeans_mpi_coreset_optlloyd > {self.parallelism}_coreset_optlloyd_{self.num_procs}.txt\n'
        elif self.parallelism == HYBRID and not self.coreset:
            self.job += f'mpirun ../build/kmeans_hybrid_lloyd  >> {self.parallelism}_lloyd_{self.num_procs}.txt\n'
            self.job += f'mpirun ../build/kmeans_hybrid_optlloyd  >> {self.parallelism}_optlloyd_{self.num_procs}.txt\n'
        elif self.parallelism == HYBRID and self.coreset:
            self.job += f'mpirun ../build/kmeans_hybrid_coreset_optlloyd  >> {self.parallelism}_optlloyd_coreset_{self.num_procs}.txt\n'


def write_serial(coreset=False):
    writer = JobWriter(SERIAL, MEMORY, HOURS, MINUTES, SECONDS, coreset=coreset)
    if coreset:
        writer.write('serial_coreset.sh')
    else:
        writer.write('serial.sh')


def write_omp(coreset=False):
    for num_threads in range(2, 18, 2):
        writer = JobWriter(OMP, MEMORY, HOURS, MINUTES, SECONDS, num_threads=num_threads, coreset=coreset)
        if coreset:
            writer.write(f'omp_coreset_{num_threads}.sh')
        else:
            writer.write(f'omp_{num_threads}.sh')


def write_mpi(coreset=False):
    for num_procs in list(range(2, 18, 2)) + [32]:
        writer = JobWriter(MPI, MEMORY, HOURS, MINUTES, SECONDS, num_procs=num_procs, coreset=coreset)
        if coreset:
            writer.write(f'mpi_coreset_{num_procs}.sh')
        else:
            writer.write(f'mpi_{num_procs}.sh')


def write_hybrid(coreset=False):
    for num_procs in range(2, 5):
        writer = JobWriter(HYBRID, MEMORY, HOURS, MINUTES, SECONDS, num_procs=num_procs, coreset=coreset)
        if coreset:
            writer.write(f'hybrid_coreset_{num_procs}.sh')
        else:
            writer.write(f'hybrid_{num_procs}.sh')


if __name__ == "__main__":
    write_serial()
    write_omp()
    write_mpi()
    write_hybrid()
    write_serial(coreset=True)
    write_omp(coreset=True)
    write_mpi(coreset=True)
    write_hybrid(coreset=True)
