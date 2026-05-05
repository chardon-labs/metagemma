from __future__ import annotations

import datetime
import json
import os
import shlex
import sys
from pathlib import Path
from typing import TypeAlias, cast


JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


def instance_id_text(instance: JsonObject) -> str:
    instance_id = instance.get("id")
    return "" if instance_id is None else str(instance_id)


def existing_config_instance_id() -> str | None:
    config_path = os.environ.get("VASTAI_INSTANCE_CONFIG")
    if not config_path or not os.path.exists(config_path):
        return None

    with open(config_path, encoding="utf-8") as config_file:
        for line in config_file:
            key, separator, value = line.strip().partition("=")
            if separator and key == "VASTAI_INSTANCE_ID":
                return shlex.split(value)[0] if value else ""
    return None


def instance_ssh_target(instance: JsonObject) -> tuple[str, str]:
    host = instance.get("public_ipaddr")
    if not isinstance(host, str) or not host:
        host = instance.get("ssh_host")
    if not isinstance(host, str) or not host:
        instance_id = instance_id_text(instance) or "<unknown>"
        raise SystemExit(f"Vast.ai instance {instance_id} is missing public_ipaddr and ssh_host")

    port = None
    ports = instance.get("ports")
    if isinstance(ports, dict):
        ssh_ports = ports.get("22/tcp")
        if isinstance(ssh_ports, list):
            for binding in ssh_ports:
                if not isinstance(binding, dict):
                    continue
                host_port = binding.get("HostPort")
                if isinstance(host_port, str) and host_port:
                    port = host_port
                    break

    ssh_port = instance.get("ssh_port")
    if port is None:
        if isinstance(ssh_port, int):
            port = str(ssh_port)
        elif isinstance(ssh_port, str) and ssh_port:
            port = ssh_port

    if port is None:
        instance_id = instance_id_text(instance) or "<unknown>"
        raise SystemExit(f"Vast.ai instance {instance_id} is missing an SSH port")

    return host, port


def describe_instance(instance: JsonObject) -> str:
    instance_id = instance_id_text(instance) or "<unknown>"
    status = instance.get("actual_status") or instance.get("cur_state") or "unknown"
    gpu_name = instance.get("gpu_name") or "unknown GPU"
    geolocation = instance.get("geolocation") or instance.get("country_code") or "unknown location"
    host, port = instance_ssh_target(instance)
    return f"{instance_id} | {status} | {gpu_name} | {geolocation} | root@{host}:{port}"


def parse_instances() -> list[JsonObject]:
    instances_payload = json.loads(os.environ["VASTAI_INSTANCES_JSON"])
    if not isinstance(instances_payload, list):
        raise SystemExit("vastai show instances --raw did not return a JSON list")
    if not instances_payload:
        raise SystemExit("No Vast.ai instances found")

    instances: list[JsonObject] = []
    for instance in instances_payload:
        if not isinstance(instance, dict):
            raise SystemExit("Vast.ai instance entry is not a JSON object")
        instances.append(cast(JsonObject, instance))
    return instances


def match_instance(instances: list[JsonObject], selected_id: str) -> JsonObject | None:
    for instance in instances:
        if instance_id_text(instance) == selected_id:
            return instance
    return None


def select_instance(instances: list[JsonObject]) -> JsonObject:
    if len(instances) == 1:
        return instances[0]

    selected_id = os.environ.get("VASTAI_INSTANCE_ID")
    if selected_id:
        selected = match_instance(instances, selected_id)
        if selected is not None:
            return selected
        raise SystemExit(f"VASTAI_INSTANCE_ID={selected_id} was not found in {len(instances)} Vast.ai instances")

    existing_id = existing_config_instance_id()
    stdin_is_tty = sys.stdin.isatty()
    stdout_is_tty = sys.stdout.isatty()
    if not stdin_is_tty or not stdout_is_tty:
        if existing_id:
            selected = match_instance(instances, existing_id)
            if selected is not None:
                return selected

        sys.stderr.write("Multiple Vast.ai instances found:\n")
        for index, instance in enumerate(instances, start=1):
            sys.stderr.write(f"  {index}. {describe_instance(instance)}\n")
        raise SystemExit("Set VASTAI_INSTANCE_ID to choose an instance in non-interactive mode")

    print("Multiple Vast.ai instances found:")
    default_index = None
    for index, instance in enumerate(instances, start=1):
        current_marker = ""
        if existing_id and instance_id_text(instance) == existing_id:
            default_index = index
            current_marker = " [current]"
        print(f"  {index}. {describe_instance(instance)}{current_marker}")

    default_suffix = f" [{default_index}]" if default_index is not None else ""
    choice = input(f"Choose instance number{default_suffix}: ").strip()
    if choice == "" and default_index is not None:
        return instances[default_index - 1]
    if not choice.isdigit():
        raise SystemExit(f"Invalid instance selection: {choice!r}")

    index = int(choice)
    if index < 1 or index > len(instances):
        raise SystemExit(f"Instance selection {index} is outside 1..{len(instances)}")
    return instances[index - 1]


def write_config(config_path: Path, instance: JsonObject) -> None:
    instance_id = instance.get("id")
    status = instance.get("actual_status") or instance.get("cur_state") or "unknown"
    host, port = instance_ssh_target(instance)
    updated_at = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
    values = {
        "VASTAI_REMOTE_HOST": f"root@{host}",
        "VASTAI_REMOTE_PORT": port,
        "VASTAI_INSTANCE_ID": "" if instance_id is None else str(instance_id),
        "VASTAI_INSTANCE_STATUS": str(status),
        "VASTAI_INSTANCE_UPDATED_AT": updated_at,
    }

    with open(config_path, "w", encoding="utf-8") as config_file:
        config_file.write("# Generated by scripts/remote/update_remote_instance.sh; do not edit.\n")
        for key, value in values.items():
            config_file.write(f"{key}={shlex.quote(value)}\n")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: update_remote_instance.py <output-config-path>")
    write_config(Path(sys.argv[1]), select_instance(parse_instances()))


if __name__ == "__main__":
    main()
