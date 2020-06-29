import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from collections import defaultdict
import numpy as np
from scipy.stats import sem

RESULT_DIR = 'data'
SERIAL = 'serial'
OMP = 'omp'
MPI = 'mpi'
HYBRID = 'hybrid'
LLOYD = 'lloyd'
OPTLLOYD = 'optlloyd'
METHOD_MAP = {LLOYD: 'Lloyd',
              OPTLLOYD: 'Optimized Lloyd'}
SERIAL_COLOR1 = 'orange'
SERIAL_COLOR2 = 'red'
SERIAL_COLOR3 = 'pink'
PARALLEL_COLOR1 = 'blue'
PARALLEL_COLOR2 = 'green'
PARALLEL_COLOR3 = 'purple'
Y_SEC_LABEL = 'Time (s)'
Y_MS_LABEL = 'Time (ms)'


def parse_times():
    times = defaultdict(list)
    for filename in os.listdir(RESULT_DIR):
        with open(os.path.join(RESULT_DIR, filename), 'r', encoding='windows-1252') as file:
            for line in file.readlines():
                if 'wall' in line:
                    line = line.split('s')  # s for seconds appended to time in seconds
                    time = float(line[0].strip(' '))
                    times[filename].append(time)

    return times


def plot_serial(num_procs, times, method, color, coreset=False):
    _, y_vals, _ = get_x_and_y(times, SERIAL, method, coreset)
    y_vals, _ = scale(y_vals)
    plt.plot([1] + list(num_procs), [y_vals for _ in range(1 + len(num_procs))], color=color)


def scale(times):
    if all([True if time < 1.5 else False for time in times]):
        return [time * 1000 for time in times], 'ms'
    else:
        return times, 's'


def plot_times(times, parallelism, method, s_color, p_color, coreset=False):
    x_vals, y_vals, errors = get_x_and_y(times, parallelism, method, coreset)
    y_vals, time_scale = scale(y_vals)
    if time_scale == 'ms':
        errors, _ = scale(errors)
    labels = [1] + list(x_vals)
    if parallelism == MPI:
        x_vals = list(range(2, len(x_vals) + 2))
        plt.xticks([1] + x_vals, labels=labels)
    else:
        plt.xticks(labels)
    plot_serial(x_vals, times, method, s_color, coreset)
    result = plt.scatter(x_vals, y_vals, color=p_color, cmap=p_color)
    plt.errorbar(x_vals, y_vals, yerr=errors, c=result.get_facecolor()[0])
    plt.ylabel(Y_SEC_LABEL if time_scale == 's' else Y_MS_LABEL)

    if parallelism == OMP:
        plt.xlabel('Threads')
    elif parallelism == MPI:
        plt.xlabel('Procs')
    elif parallelism == HYBRID:
        plt.xlabel('Nodes')

    serial = mlines.Line2D([], [], color=s_color, linestyle='-',
                           markersize=15, label=SERIAL.capitalize() + '_' + method.capitalize())
    parallel = mlines.Line2D([], [], color=result.get_facecolor()[0], linestyle='-',
                             markersize=15, label=parallelism.upper() + '_' + method.capitalize())
    return serial, parallel


def get_x_and_y(times, parallelism, method, coreset=False):
    x_vals = []
    y_vals = []
    errors = []
    for key, value in times.items():
        if coreset and 'coreset' not in key:
            continue
        elif not coreset and 'coreset' in key:
            continue

        method_name = parse_method(key)
        if parallelism in key and method == method_name:
            num_procs = key.split('_')[-1].split('.')[0]
            try:
                x_vals.append(int(num_procs))
                y_vals.append(np.average(value))
                errors.append(sem(value))
            except ValueError:
                x_vals.append(1)
                y_vals.append(value[0])
                errors.append(0)

    x_vals, y_vals, errors = zip(*sorted(zip(x_vals, y_vals, errors), key=lambda x: x[0]))
    return x_vals, y_vals, errors


def parse_method(filename):
    filename = filename.split('_')
    if len(filename) == 3 and 'coreset' not in filename:
        return filename[1]
    elif len(filename) == 2:
        return filename[-1].split('.')[0]
    elif len(filename) == 3 and 'coreset' in filename:
        return filename[-1].split('.')[0]
    elif len(filename) == 4 and 'coreset' in filename:
        return filename[2]


if __name__ == "__main__":
    times = parse_times()

    # opt_serial, opt_parallel = plot_times(times, OMP, OPTLLOYD, SERIAL_COLOR2, PARALLEL_COLOR2)
    # opt_coreset_serial, opt_coreset_parallel = plot_times(
    #     times, OMP, OPTLLOYD, SERIAL_COLOR1, PARALLEL_COLOR1, coreset=True)

    opt_serial, opt_parallel = plot_times(times, MPI, OPTLLOYD, SERIAL_COLOR2, PARALLEL_COLOR2)
    # opt_coreset_serial, opt_coreset_parallel = plot_times(
    #     times, MPI, OPTLLOYD, SERIAL_COLOR1, PARALLEL_COLOR1, coreset=True)

    # opt_serial, opt_parallel = plot_times(times, HYBRID, OPTLLOYD, SERIAL_COLOR2, PARALLEL_COLOR2)
    # opt_coreset_serial, opt_coreset_parallel = plot_times(
    #     times, HYBRID, OPTLLOYD, SERIAL_COLOR1, PARALLEL_COLOR1, coreset=True)

    plt.legend(handles=[opt_serial, opt_parallel], loc='center', bbox_to_anchor=(0.825, 0.75))
    # plt.legend(handles=[opt_coreset_serial, opt_coreset_parallel], loc='center', bbox_to_anchor=(0.825, 0.75))
    plt.title('MPI OptLloyd KMeans')
    plt.tight_layout()
    # plt.show()
    plt.savefig('mpi_optlloyd.png', dpi=1400)
