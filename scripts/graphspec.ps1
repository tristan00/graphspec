param(
    [Parameter(Mandatory = $true)]
    [Alias('Home')]
    [string]$RuntimeHome,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Command
)

$ErrorActionPreference = 'Stop'
if ($RuntimeHome -notmatch '^([A-Za-z]:[\\/]|\\\\)') {
    throw 'Home must be an absolute path outside the code checkout'
}
$env:GRAPHSPEC_HOME = (Resolve-Path -LiteralPath $RuntimeHome).ProviderPath
$env:TEMP = Join-Path $env:GRAPHSPEC_HOME 'tmp'
$env:TMP = $env:TEMP
$env:TMPDIR = $env:TEMP
$env:PYTHONDONTWRITEBYTECODE = '1'
$RuntimePython = Join-Path $env:GRAPHSPEC_HOME 'environment/Scripts/python.exe'
& $RuntimePython -B -m graphspec @Command
exit $LASTEXITCODE
