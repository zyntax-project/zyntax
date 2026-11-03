# CI benchmark machines

The benchmark pages on the website (`/benchmark`, `/zylua`, `/zypy`) are built by the `Publish Wiki & Deploy Site` workflow (`.github/workflows/wiki.yml`) on GitHub-hosted runners. A runner is a fresh virtual machine, and the machine behind a given label is not fixed, so a result only means something next to a description of the machine that produced it. This page says what GitHub documents about the runners, what it leaves out, and what each result records instead.

## What GitHub documents

From [GitHub-hosted runners reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners), standard runners in a public repository:

| label | vCPU | RAM | disk | architecture |
|---|---|---|---|---|
| `ubuntu-latest` | 4 | 16 GB | 14 GB SSD | x64 |
| `windows-latest` | 4 | 16 GB | 14 GB SSD | x64 |
| `macos-latest` | 3 (M1) | 7 GB | 14 GB SSD | arm64 |
| `macos-15-intel` | 4 | 14 GB | 14 GB SSD | Intel |

Private repositories get 2 vCPUs and 8 GB on the Linux and Windows labels. These figures change; the page above is the source.

From [About GitHub-hosted runners](https://docs.github.com/en/actions/concepts/runners/github-hosted-runners):

- Each runner is a new virtual machine.
- Standard 2-core and 4-core Linux runners may run on multi-cloud compute, which uses AWS or Azure. Other Linux runners and the Windows runners run on Azure VMs. macOS runners run in Azure data centers.

## What GitHub does not document

The CPU model, the hypervisor, whether a vCPU is a hardware thread or a core, cache sizes, clock speed, and whether performance is the same from run to run. The documentation gives no command for finding them. Do not assume them: two runs of the same job on `ubuntu-latest` have run on different CPU vendors.

## What each result records

Every language benchmark result file carries a `meta` block written by the `Record run metadata` step:

| field | source |
|---|---|
| `cpu` | `model name` in `/proc/cpuinfo`, or `machdep.cpu.brand_string` on macOS |
| `cpu_cores`, `ram_gb` | `getconf _NPROCESSORS_ONLN`, `/proc/meminfo` |
| `hardware.vendor`, `hardware.product` | `/sys/class/dmi/id/sys_vendor` and `product_name`; these name the cloud (and, on some clouds, the instance type) |
| `hardware.hypervisor` | `lscpu`, `Hypervisor vendor` |
| `hardware.threads_per_core`, `hardware.cores_per_socket` | `lscpu`; whether the vCPUs the job sees are hardware threads |
| `hardware.l2`, `hardware.l3` | `lscpu`; empty where the machine does not report them |
| `commit`, `date` | the commit and time of the run |

Results from before the `hardware` fields were added have only the first rows.

## Reading the pages

The Lua and Python pages keep one tab for each machine CI has measured. A run measures only the machines it was given, so the deploy carries every other machine's last result forward from the site published before (`.github/scripts/carry_machines.py`, files under `<page>/machines/`, listed in `machines/index.json`). The page opens on the machine measured last. A tab that has not been measured for a while still shows its own commit and date, so a stale tab is visible.

- Below the table, **Every machine, last five runs** has one row per machine and one column: the geometric mean, over that machine's last five runs, of how much faster zylua is than LuaJIT (zypy than PyPy) across the benchmarks, with an **All machines** row that counts those runs once each. It follows the Benchmark time / Full process switch. Only recent runs count because the code changes between runs. Each run's figure is stored in `machines/history.json`, which keeps every run and is carried forward at every deploy; the first deploy started from the runs in `.github/scripts/history-seed/`, taken from the artifacts of the runs then available. A geometric mean lets a few large wins outweigh many small losses, so read it beside the win count above the main table.
- Compare within a tab. A ratio against PyPy or LuaJIT depends on the machine, because the baseline interpreter does not run equally well on every CPU: in the runs measured so far PyPy ran far slower on the Intel runners than on the AMD ones while zypy did not, so the same code won more kernels on the Intel tabs.
- Do not compare absolute times between tabs of different CPUs.
- One tab is the latest run on that machine, so run-to-run variation on the same CPU model is not visible in it.
- The ZynML kernel page (`/benchmark`) has not been split by machine. Its kernels run as separate jobs that can each land on a different CPU, so one result file may mix machines; each partial carries its own `meta.cpu`.

## Getting the data

Each run uploads its result files as artifacts:

```sh
gh run list --workflow wiki.yml
gh run download <run id> -n lang-zylua-ubuntu-latest   # zylua.json
gh run download <run id> -n lang-zypy-ubuntu-latest    # zypy.json
```

`meta.cpu` in the file says which machine produced it. Artifacts are kept for 90 days.
