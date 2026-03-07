# TCP Protocol (Camera Device <-> Server)

Each packet is one JSON object terminated by `\n`.

## Device -> Server

1. Register

```json
{"type":"register","device_code":"00000001","ts":1700000000}
```

1. Heartbeat

```json
{"type":"heartbeat","device_code":"00000001","cpu_temp_c":52.3,"cpu_usage":31.5,"mem_usage":48.2,"ts":1700000001}
```

1. Person Appeared Event

```json
{"type":"event","event":"person_appeared","person_id":12,"ts":1700000002}
```

1. Capture Complete Event

```json
{"type":"event","event":"capture_complete","person_id":12,"image_type":"face","ts":1700000003}
```

## Server -> Device

1. Sleep

```json
{"type":"command","cmd":"sleep"}
```

Meaning: enter low-power mode, stop camera/inference tasks.

1. Wake

```json
{"type":"command","cmd":"wake"}
```

Meaning: restart camera/inference tasks.

1. Capture

```json
{"type":"command","cmd":"capture"}
```

Meaning: trigger immediate snapshot.

1. Config Update

```json
{
  "type":"config_update",
  "cmd":"config_update",
  "config":{
    "device":{"code":"12345678"},
    "upload":{"server":"http://192.168.1.2:11100","image_path":"/receive/image/auto"},
    "tcp":{"server_ip":"192.168.1.1","port":19000,"heartbeat_interval_sec":10,"reconnect_interval_sec":3}
  }
}
```

Meaning: device applies config and writes to `device_config.json`.
