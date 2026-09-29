param(
    [Parameter(Mandatory = $true)]
    [Alias('Home')]
    [string]$RuntimeHome,
    [Parameter(Mandatory = $true)]
    [string]$Log,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Command
)

$ErrorActionPreference = 'Stop'
foreach ($Path in @($RuntimeHome, $Log)) {
    if ($Path -notmatch '^([A-Za-z]:[\\/]|\\\\)') {
        throw "$Path must be an absolute path"
    }
}
$RuntimeHome = (Resolve-Path -LiteralPath $RuntimeHome).ProviderPath
$Temp = Join-Path $RuntimeHome 'tmp'
$Python = Join-Path $RuntimeHome 'environment\Scripts\python.exe'
$Arguments = ($Command | ForEach-Object { if ($_ -match '[\s"]') { '"' + ($_ -replace '"', '\"') + '"' } else { $_ } }) -join ' '
$Settings = "set `"GRAPHSPEC_HOME=$RuntimeHome`" && set `"TEMP=$Temp`" && set `"TMP=$Temp`" && set `"TMPDIR=$Temp`" && set PYTHONDONTWRITEBYTECODE=1"
$Line = "cmd.exe /d /c `"$Settings && `"$Python`" -B -m tuning $Arguments > `"$Log`" 2>&1`""
$Result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = $Line
    CurrentDirectory = (Split-Path $PSScriptRoot -Parent)
}
if ($Result.ReturnValue -ne 0) {
    throw "process creation failed with code $($Result.ReturnValue)"
}
Write-Output ("started pid={0} at={1:o} log={2}" -f $Result.ProcessId, (Get-Date), $Log)
