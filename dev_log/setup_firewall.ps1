$programs = @(
    "C:\SIMULIA\Commands\abaqus.bat",
    "C:\SIMULIA\Commands\abaqus.exe",
    "C:\SIMULIA\CAE\2024LE\win_b64\code\bin\standard.exe",
    "C:\SIMULIA\CAE\2024LE\win_b64\code\bin\explicit.exe",
    "C:\SIMULIA\CAE\2024LE\win_b64\code\bin\ABQcaeK.exe",
    "C:\SIMULIA\CAE\2024LE\win_b64\code\bin\ABQmain.exe"
)

foreach ($prog in $programs) {
    if (Test-Path $prog) {
        $ruleNameIn = "BlockIn_Abaqus_" + (Split-Path $prog -Leaf)
        $ruleNameOut = "BlockOut_Abaqus_" + (Split-Path $prog -Leaf)
        
        # Remove existing rules to prevent duplicates
        Remove-NetFirewallRule -DisplayName $ruleNameIn -ErrorAction SilentlyContinue
        Remove-NetFirewallRule -DisplayName $ruleNameOut -ErrorAction SilentlyContinue
        
        # Add new rules
        New-NetFirewallRule -DisplayName $ruleNameIn -Direction Inbound -Program $prog -Action Block -Profile Any
        New-NetFirewallRule -DisplayName $ruleNameOut -Direction Outbound -Program $prog -Action Block -Profile Any
        Write-Host "Blocked network access for $prog"
    } else {
        Write-Host "Path not found, skipping $prog"
    }
}
