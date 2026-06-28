$exes = @(
    "C:\SIMULIA\CAE\2024LE\win_b64\code\bin\LE\standard.exe",
    "C:\SIMULIA\CAE\2024LE\win_b64\code\bin\LE\explicit.exe",
    "C:\SIMULIA\CAE\2024LE\win_b64\code\bin\LE\ABQcaeK.exe"
)

foreach ($exe in $exes) {
    if (Test-Path $exe) {
        $name = [System.IO.Path]::GetFileNameWithoutExtension($exe)
        # Block Outbound
        New-NetFirewallRule -DisplayName "Block Abaqus Outbound ($name)" -Direction Outbound -Program $exe -Action Block -ErrorAction SilentlyContinue
        # Block Inbound
        New-NetFirewallRule -DisplayName "Block Abaqus Inbound ($name)" -Direction Inbound -Program $exe -Action Block -ErrorAction SilentlyContinue
        Write-Host "Successfully Blocked Network for: $exe"
    } else {
        Write-Host "Executable Not Found: $exe"
    }
}
Write-Host "Abaqus offline network isolation applied."
