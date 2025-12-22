
use std::cell::RefCell;

use proc_macro::TokenStream;
use quote::{format_ident, quote, ToTokens};
use syn::{parenthesized, parse::{Parse, ParseStream}, parse_macro_input, parse_quote, punctuated::Punctuated, spanned::Spanned, visit::Visit, Expr, Ident, ItemImpl, Token, Type};
use syn::Result;


#[proc_macro_attribute]
pub fn routines(attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut impl_block = parse_macro_input!(item as ItemImpl);
    let generics: Vec<_> = impl_block.generics.type_params().map(|p| p.ident.clone()).collect();

    let type_conversions = parse_macro_input!(attr as RpcArgs);
    // goal: find all impl blocks (they will be empty and unimplemented)
    // then, take the name in the stayle of name_of_method(<&self or &mut self>, name: Type, name2: Type2) -> ReturnType { ... }
    // we take that, and make an enum variant:
    // NameOfMethod { name: Type, name2: Type2 } 
    // NameOfMethodResponse(ReturnType)

    let mut message_variants = Vec::new();

    for item in &mut impl_block.items {
        if let syn::ImplItem::Fn(method) = item {
            let rpc_args = if let Some(attr) = rpc_attr(&method.attrs) {
                let args = RpcMethodArgs::parse(attr).expect("Failed to parse args");
                method.attrs.retain(|a| 
                    !a.path().is_ident("rpc")
                ); // remove helper attr
                Some(args)
            } else {
                None
            }.unwrap_or_default();
            if rpc_args.skip {
                continue;
            }
            let (message_signature, response_signature) = process_method(method, &generics, &type_conversions, &rpc_args);
            message_variants.push(message_signature.variant);
            message_variants.push(response_signature.variant);
            
            let mut new_method = method.clone();
            // let message_args = message_signature.initizliers.iter().map(|name| {
            //     quote! {
            //         #name: #name.into()
            //     }
            // });
            let variant_name = &message_signature.name;
            let initializers = &message_signature.initizliers;
            let response_name = &response_signature.name;
            new_method.block = parse_quote!({
                let message = RpcMessages::#variant_name {
                    #( #initializers ),*
                };
                // send_recv!(self, message, RpcMessages::#response_signature(result) => {
                //     result?.into()
                // })
                let receiver = self.send_message(message);
                let response = receiver.recv()
                    .map_err(|e| TensorError::BackendError(format!("Failed to receive RPC response: {}", e)))?;
                match response {
                    RpcMessages::#response_name(result) => {
                        result?.into()
                    },
                    _ => {
                        return Err(TensorError::BackendError("Received unexpected RPC response message".to_string()));
                    }
                }
                // panic!()
            });

            *item = syn::ImplItem::Fn(new_method);
        }
    }
    let message_variants = quote::quote! {
        #impl_block
        pub(crate) mod rpc_proc {
            use super::*;
            pub(crate) enum RpcMessages {
                #( #message_variants )*
            }
        }
    };

    // now, we need to remake the impl_block, implementing each method to send the appropriate message

    message_variants.into()
}

struct MessageSignature {
    variant: proc_macro2::TokenStream,
    name: Ident,
    initizliers: Vec<proc_macro2::TokenStream>,
}

fn process_method(
    method: &mut syn::ImplItemFn, 
    generics: &[Ident],
    type_conversions: &RpcArgs,
    rpc_args: &RpcMethodArgs
) -> (MessageSignature, MessageSignature) {

    let method_name = &method.sig.ident;
    let variant_name = format_ident!("{}", from_snake_to_camel(&method_name.to_string()));
    let response_variant_name = format_ident!("{}Response", variant_name);
    

    let initizliers = RefCell::new(Vec::new());
    let fields = method.sig.inputs.iter().filter_map(|arg| {
        if let syn::FnArg::Typed(pat) = arg {
            let name = &pat.pat;
            let ty = &pat.ty;
            let ty_mapped = map_types(ty, &type_conversions);
            if type_uses_generics(&ty_mapped, &generics) {
                return Some(syn::Error::new(ty.span(), "Generic types in RPC method arguments are not supported").to_compile_error());
            }
            // args.borrow_mut().push(stream_to_type(&name.to_token_stream()));
            let init = if let syn::Type::Tuple(tuple_type) = &ty_mapped {
                // each elemement is into
                // so (src.0.into(), src.1.into(), etc....)
                let inits = tuple_type.elems.iter().enumerate().map(|(i, _)| {
                    let index = syn::Index::from(i);
                    quote::quote! {
                        #name.#index.into()
                    }
                });
                quote::quote! {
                    ( #( #inits ),* )
                }
            } else{
                quote::quote! {
                    #name.into()
                }
            };
            initizliers.borrow_mut().push(quote::quote! {
                #name: #init
            });
            Some(quote::quote! {
                #name: #ty_mapped
            })
        } else {
            None
        }
    });

    let fields = fields.chain(rpc_args.extra.iter().map(|arg| {
        let name = &arg.name;
        let ty = &arg.ty;
        // args.borrow_mut().push(stream_to_type(&name.to_token_stream()));
        let init_arg = &arg.extract_expr;
        initizliers.borrow_mut().push(quote::quote! {
            #name: #init_arg
        });
        quote::quote! {
            #name: #ty
        }
    }));

    let resp_field = match &method.sig.output {
        syn::ReturnType::Type(_, ty) => {
            // replace types
            let ty_mapped = map_types(ty, &type_conversions);
            if type_uses_generics(&ty_mapped, &generics) {
                syn::Error::new(ty.span(), "Generic types in RPC method return types are not supported").to_compile_error()
            }else {
                quote::quote! {#ty_mapped}
            }
        },
        syn::ReturnType::Default => {
            quote::quote! {()}
        }
    }; 
    let signature = MessageSignature {
        variant: quote::quote! {
            #variant_name { #( #fields ),* },
        },
        name: variant_name.clone(),
        // args: args.borrow().clone(),
        initizliers: initizliers.borrow().clone(),
    };
    let response_signature = MessageSignature {
        variant: quote::quote! { 
            #response_variant_name ( #resp_field ),
        },
        name: response_variant_name.clone(),
        // args: vec![],
        initizliers: vec![],
    };
    (signature, response_signature)
}

#[derive(Debug)]
struct TypeMapping {
    from: syn::Type,
    to: syn::Type,
}

impl Parse for TypeMapping {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let from: syn::Type = input.parse()?;
        input.parse::<syn::Token![=]>()?;
        let to: syn::Type = input.parse()?;
        Ok(TypeMapping { from, to })
    }
}

#[derive(Debug)]
struct RpcArgs {
    mappings: Punctuated<TypeMapping, Token![,]>,
}

impl Parse for RpcArgs {
    fn parse(input: ParseStream) -> Result<Self> {
        Ok(RpcArgs {
            mappings: input.parse_terminated(TypeMapping::parse, Token![,])?,
        })
    }
}

#[inline]
fn from_snake_to_camel(s: &str) -> String {
    s.split('_')
        .map(|word| {
            let mut c = word.chars();
            match c.next() {
                None => String::new(),
                Some(f) => f.to_uppercase().collect::<String>() + &c.as_str(),
            }
        })
        .collect()
}

fn type_equal_tokens(a: &syn::Type, b: &syn::Type) -> bool {
    quote::quote! { #a }.to_string() == quote::quote! { #b }.to_string()
}


fn map_types(ty: &syn::Type, mappings: &RpcArgs) -> syn::Type {
    for mapping in &mappings.mappings {
        if type_equal_tokens(ty, &mapping.from) {
            return mapping.to.clone();
        }
    }
    ty.clone()
}
struct GenericDetector<'a> {
    generics: &'a [Ident],
    found: bool,
}

impl<'a, 'ast> Visit<'ast> for GenericDetector<'a> {
    fn visit_ident(&mut self, i: &'ast Ident) {
        if self.generics.iter().any(|g| g == i) {
            self.found = true;
        }
    }
}

fn type_uses_generics(ty: &Type, generics: &[Ident]) -> bool {
    let mut v = GenericDetector { generics, found: false };
    v.visit_type(ty);
    v.found
}

#[derive(Debug)]
struct ExtraMethodArg {
    name: Ident,
    ty: Type,
    extract_expr: Expr
}

impl Parse for ExtraMethodArg {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let name: Ident = input.parse()?;
        input.parse::<Token![:]>()?;
        let ty: Type = input.parse()?;
        input.parse::<Token![=]>()?;
        let extract_expr: Expr = input.parse()?;
        Ok(ExtraMethodArg { name, ty, extract_expr })
    }
}

struct RpcMethodArgs {
    skip: bool,
    sync: bool,
    extra: Vec<ExtraMethodArg>,
}

impl Default for RpcMethodArgs {
    fn default() -> Self {
        Self {
            skip: false,
            sync: true,
            extra: vec![],
        }
    }
}

impl RpcMethodArgs {
    fn parse(attr: &syn::Attribute) -> syn::Result<Self> {
        let mut args = RpcMethodArgs::default();
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("skip") {
                args.skip = true;
                Ok(())
            } else if meta.path.is_ident("extra") {
                // let parsed = meta.input.parse_terminated(ExtraMethodArg::parse, Token![,])?;
                let content;
                parenthesized!(content in meta.input);
                let parsed = content.parse_terminated(ExtraMethodArg::parse, Token![,])?;
                args.extra.extend(parsed);
                Ok(())
            } else if meta.path.is_ident("sync") {
                args.sync = true;
                Ok(())
            }  else {
                Err(meta.error("unsupported rpc attribute argument"))
            }
        })?;

        Ok(args)
    }
}

fn rpc_attr(attrs: &[syn::Attribute]) -> Option<&syn::Attribute> {
    attrs.iter().find(|a| a.path().is_ident("rpc"))
}

fn stream_to_type(ts: &proc_macro2::TokenStream) -> syn::Ident {
    syn::parse(ts.clone().into()).expect("Failed to parse type from TokenStream")
}