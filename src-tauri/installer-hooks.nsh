; Put the rt-captions CLI on the user's PATH on install; take it off on uninstall.
!macro NSIS_HOOK_POSTINSTALL
  nsExec::Exec '"$INSTDIR\rt-captions.exe" install-cli'
!macroend

!macro NSIS_HOOK_PREUNINSTALL
  nsExec::Exec '"$INSTDIR\rt-captions.exe" uninstall-cli'
!macroend
