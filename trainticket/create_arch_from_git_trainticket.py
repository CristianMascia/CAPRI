import json
import os


def __main__(path_git, path_arch):
    services = [s for s in os.listdir(path_git) if "ts-" in s]

    with open(path_arch, 'w') as f_arch:
        arch = {}

        for ser in services:
            if ser in ['ts-gateway-service', 'ts-ticket-office-service', 'ts-avatar-service', 'ts-ui-service',
                       'ts-ui-dashboard', 'ts-news-service', 'ts-delivery-service', 'ts-common']:
                continue

            if ser == "ts-voucher-service":
                arch[ser] = ['ts-order-service', 'ts-order-other-service']
            else:
                path = os.path.join(path_git, ser)
                impls = []
                for (dir_path, dir_names, file_names) in os.walk(path):
                    for f in file_names:
                        if "Impl.java" in f:
                            impls.append(os.path.join(dir_path, f))

                if len(impls) == 0:
                    print("ERRORE in " + ser)
                    continue

                list_called = []
                for impl in impls:
                    with open(impl, 'r') as f_f:
                        print("--------FOR SERVICE: " + ser + "----------")
                        for line in f_f.readlines():
                            if "getServiceUrl(\"ts" in line:
                                ser_called = line[line.rfind('(\"') + 2:line.rfind('"')]
                                list_called.append(ser_called)
                    if len(list_called) > 0:
                        list_called = list(set(list_called))
                        print(list_called)
                        arch[ser] = list_called
                    else:
                        print("NO CALL")
        json.dump(arch, f_arch)
