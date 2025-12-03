$base = "C:\Users\juhe9\repos\MasterThesis\ForkSight\Segmentation\Data"
$remoteimages = "cl-tars:/data/jhehli/raw_data/images"
$remotemasks = "cl-tars:/data/jhehli/raw_data/masks"

$images = Join-Path $base "images"
$masks  = Join-Path $base "masks"

Write-Host "Uploading masks..."
scp (Join-Path $masks "*") $remotemasks

Write-Host "Uploading images with matching masks..."
Get-ChildItem $images | ForEach-Object {
    $img = $_.FullName
    $filename = $_.Name
    $maskPath = Join-Path $masks $filename
    
    if (Test-Path $maskPath) {
        scp $img $remoteimages
    }
}

Write-Host "Done!"
