# Pump P-100 Service Manual

## Overview

The P-100 is a centrifugal seawater cooling pump rated at 250 kW. It is fitted to all Coastguard patrol vessels and serves the main engine cooling jacket loop. Operating pressure is 4.5 bar nominal, 6.0 bar peak. Maximum continuous shaft speed is 1450 rpm.

## Alarm Codes

The pump controller exposes alarm codes via the vessel's CAN bus. Codes are 2-character hexadecimal values.

### Code 4F — Suction-side low pressure

Cause: air ingestion at the suction strainer, blocked sea chest, or sea-state induced surface skim. This code is generated when suction pressure falls below 0.4 bar for more than 3 seconds.

Recommended actions, in order:
1. Confirm sea chest valve is fully open.
2. Inspect the suction strainer for debris. The strainer is rated for 2mm mesh; biological fouling is common.
3. Check for air entrainment from a partially-submerged sea chest in heavy swell. If the alarm correlates with vessel pitch angle, the cause is hydrodynamic, not mechanical.
4. If the strainer is clean and suction is fully submerged, suspect a leaking suction-side gasket. Pressurize the line and inspect.

Code 4F suppression: do not suppress this alarm without isolating cause. Cavitation damage to the impeller can occur within 20 minutes of sustained low-suction operation.

### Code 6A — High bearing temperature

Cause: bearing wear, lubrication failure, or coupling misalignment. Trips at 85°C; pre-alarm at 75°C.

Recommended actions:
1. Verify lubricant level in the bearing housing sight glass.
2. If lubricant is present and clear, check for coupling misalignment using a dial indicator. Maximum permissible misalignment is 0.15mm radial.
3. Vibration analysis should be performed using portable accelerometer; characteristic 1x running-speed peak indicates imbalance, 2x indicates misalignment.

### Code 7C — Vibration trip

Cause: bearing failure, impeller damage, or cavitation. Trips at 12 mm/s RMS velocity in any axis.

Stop the pump. Do not reset. Inspect impeller for erosion damage before restart.

## Routine Maintenance Schedule

- Daily: visual inspection, leak check at flange joints.
- Weekly: strainer cleaning at sea chest.
- 250 hours: lubricant top-up, vibration baseline reading.
- 1000 hours: bearing inspection, alignment check, impeller inspection.
- 5000 hours: full overhaul including mechanical seal replacement.

## Spare Parts

Recommended onboard spares for one P-100:
- Mechanical seal cartridge, P/N MSC-100-A
- Bearing set (drive end + non-drive end), P/N BRG-100-PR
- Impeller, P/N IMP-100-B (bronze, standard for seawater)
- Suction-side gasket kit, P/N GSK-100-S

## Known Issues

A batch of P-100 pumps delivered in Q2 2023 had an undersized suction-side weld bead that caused intermittent low-pressure alarms (code 4F) under heavy swell. Affected units are identified by serial numbers P100-2023-Q2-* and were retrofitted in dry-dock during the 2024 maintenance cycle.
