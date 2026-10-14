Write-Host "=== Present USB devices ==="
Get-PnpDevice -PresentOnly | Where-Object { $_.InstanceId -match '^USB' -or $_.Class -in 'Ports','USB' } |
    Select-Object Status, Class, FriendlyName, InstanceId | Format-List

Write-Host "=== Problem devices (no driver) ==="
Get-PnpDevice | Where-Object { $_.Status -eq 'Error' -or $_.Status -eq 'Unknown' -or $_.ConfigManagerErrorCode -ne 0 } |
    Select-Object Status, Class, FriendlyName, InstanceId, ConfigManagerErrorCode | Format-List

Write-Host "=== COM ports ==="
Get-CimInstance Win32_PnPEntity | Where-Object { $_.Name -match 'COM\d+' } |
    Select-Object Name, DeviceID, Status | Format-List
