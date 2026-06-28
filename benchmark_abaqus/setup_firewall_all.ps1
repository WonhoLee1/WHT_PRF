$baseDir = "C:\SIMULIA"
$allExes = Get-ChildItem -Path $baseDir -Filter *.exe -Recurse -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName

$count = 0
foreach ($exe in $allExes) {
    # Generate a safe name (max 200 chars maybe)
    $name = [System.IO.Path]::GetFileNameWithoutExtension($exe)
    $ruleNameOut = "Block_Abaqus_Out_$name`_$count"
    $ruleNameIn = "Block_Abaqus_In_$name`_$count"
    
    # Block Outbound
    New-NetFirewallRule -DisplayName "Abaqus Strict Block Out ($name)" -Name $ruleNameOut -Direction Outbound -Program $exe -Action Block -ErrorAction SilentlyContinue | Out-Null
    # Block Inbound
    New-NetFirewallRule -DisplayName "Abaqus Strict Block In ($name)" -Name $ruleNameIn -Direction Inbound -Program $exe -Action Block -ErrorAction SilentlyContinue | Out-Null
    
    $count++
}

Write-Host "Successfully applied network block rules to $count executables in $baseDir"
