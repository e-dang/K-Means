import matplotlib.pyplot as plt

plt.style.use('seaborn-darkgrid')

procs = [2, 4, 8, 16]

# kpp_mpi_times = [1077, 534, 271, 148]
coresets_mpi_times = [2.22, 0.927, 1, 0.972]

plt.figure()
# plt.plot(procs, kpp_mpi_times, label='K++')
plt.plot(procs, coresets_mpi_times, label='Coreset Sampling')
plt.plot(procs, coresets_mpi_times, label='Coreset Fitting')
plt.plot(procs, coresets_mpi_times, label='Coresets Total')
plt.legend()
plt.xticks(procs)
plt.xlabel('Number of Processes')
plt.ylabel('Run Time (s)')
plt.title('MPI Implementation 1,000,000 Datapoints')
plt.tight_layout()
plt.savefig('MPI_plot.png')
plt.show()