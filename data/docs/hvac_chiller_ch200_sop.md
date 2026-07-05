# HVAC Chiller Unit CH-200 Standard Operating Procedure

## Scope

This SOP covers operation and first-line troubleshooting of CH-200 air-cooled chiller units installed in coastal radar stations. CH-200 chillers provide 50 kW cooling capacity to the radar electronics cabinet. Refrigerant is R-410A.

## Pre-Start Checklist

Before starting a CH-200 from cold shutdown:

1. Confirm power isolation switch is in the ON position.
2. Verify chilled water loop pressure between 1.5 and 2.0 bar at the expansion tank gauge.
3. Confirm condenser fans are unobstructed. Coastal salt accumulation must be hosed off with fresh water at least weekly.
4. Check for refrigerant leak indicators on the receiver sight glass — clear sight glass with no bubbles after 10 minutes of running.

## Startup Sequence

1. Energize control panel.
2. Start chilled water pump first. Wait 60 seconds for flow stabilization.
3. Start compressor. Compressor will not engage if flow switch (FS-1) does not register flow.
4. Monitor suction pressure. Normal range: 8.5–9.5 bar.
5. Monitor discharge pressure. Normal range: 18–22 bar in 25°C ambient.

## Common Faults

### Fault E-12 — Compressor short cycling

Compressor starts and stops at intervals shorter than 5 minutes. This is a protective behavior; sustained short cycling damages the compressor.

Likely causes:
- Low refrigerant charge causing low-pressure cutout.
- Thermostat sensor placed in stagnant air, reading ambient instead of return air.
- Fouled condenser causing high-pressure cutout.

Action: do not bypass the lockout. Investigate refrigerant charge and condenser condition first.

### Fault E-18 — Low chilled water flow

Triggered by flow switch FS-1 when flow drops below 30 L/min.

Likely causes:
- Air pocket in the chilled water loop. Bleed at the high-point air vent.
- Strainer fouling at the pump inlet.
- Failed pump impeller. Listen for cavitation noise; check current draw against nameplate.

### Fault E-24 — Refrigerant leak detected

Triggered by the R-410A leak sensor mounted at the unit base.

Action: immediately shut down. Evacuate the equipment room. Refrigerant exposure is generally non-toxic but displaces oxygen in confined spaces. Do not restart until a leak inspection is complete and the cause is rectified.

## Salt Air Considerations

CH-200 units in coastal installations are subject to accelerated corrosion of condenser fins. Service interval for fin cleaning is reduced from the standard 6-month interval to 6 weeks. Use a soft-bristle brush and demineralized water spray. Do not use pressure washers directly on fins — fin damage will reduce heat transfer permanently.

Operating in salt air also accelerates electrical connector corrosion. Inspect all DIN connectors at quarterly intervals; apply dielectric grease at every reseating.

## Spare Parts

For CH-200 first-line stocking:
- Flow switch FS-1, P/N CH-FS-1
- Pressure transducer (suction + discharge variants), P/N CH-PT-S and CH-PT-D
- Condenser fan motor, P/N CH-FAN-M
- Filter-drier element, P/N CH-FDR-2

## Documentation

All maintenance activities must be logged in the station maintenance log. Refrigerant additions and removals must additionally be recorded in the F-gas log per regulatory requirements.
