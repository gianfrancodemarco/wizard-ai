from typing import Any, Dict, Optional, Type

from langchain.callbacks.manager import (AsyncCallbackManagerForToolRun,
                                         CallbackManagerForToolRun)
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from .tool_dummy_payload import ToolDummyPayload


class FormTool(BaseTool):
    # FormTool methods should take context as FormStructuredChatExecutorContext, but this creates circular references
    # So we use BaseModel instead
    def _run(
        self,
        *args: Any,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        context: Optional[BaseModel] = None,
        **kwargs
    ) -> str:
        pass

    def activate(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        context: Optional[BaseModel] = None,
    ):
        """
        Function called when the tool is activated.
        """

        pass

    async def aactivate(
        self,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        context: Optional[BaseModel] = None,
    ):
        pass

    # TODO: using context.form as dict is wrong. We need to use a Pydantic model to enable validation etc
    async def aupdate(
        self,
        *args: Any,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        context: Optional[BaseModel] = None,
    ):
        pass

    async def ais_form_complete(
        self,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        context: Optional[BaseModel] = None,
    ) -> bool:
        """
        The default implementation checks if all values except optional ones are set.
        """
        for field_name, field_info in self.args_schema.__fields__.items():
            if field_info.is_required():
                if not getattr(context.form, field_name):
                    return False
        return True

    def get_tool_start_message(self, input: dict) -> str:
        return "Creating form\n"
    
    def get_information_to_collect(self) -> str:
        return str(list(self.args.keys()))



class FormStructuredChatExecutorContext(BaseModel):
    active_form_tool: Optional[FormTool] = None
    form: BaseModel = None



class FormToolActivator(BaseTool):
    args_schema: Type[BaseModel] = ToolDummyPayload
    form_tool_class: Type[FormTool]
    form_tool: FormTool

    def _run(
        self,
        *args: Any,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        context: Optional[FormStructuredChatExecutorContext] = None,
        **kwargs
    ) -> str:
        return f"Entered in {self.form_tool.name} context"

    def _parse_input(self, tool_input: str | Dict) -> str | Dict[str, Any]:
        """FormToolActivator shouldn't have any input, so we ovveride the default implementation."""
        return {}